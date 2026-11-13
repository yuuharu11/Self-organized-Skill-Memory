"""Serve a memory-only RoboCasa policy (no π0.5 action-head inference).

The frozen π0.5 model is used solely to extract its final VLM prefix feature.
All returned action values come from retrieved variable-length memory units,
the learned magnitude head, the Context gripper head, and stored unit channels.
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

os.environ.setdefault("SKILLNET_SUBTASK_GENERATE", "0")

ROOT = Path(__file__).resolve().parents[4]
SKILLNET_SRC = ROOT / "skill_moe/skillnet/src"
EXPERT_DIR = ROOT / "skill_moe/skillnet/examples/memory_pilot/robocasa"
for _path in (SKILLNET_SRC, EXPERT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import expert as expert_api  # noqa: E402


def _network_classes():
    import torch
    from torch import nn

    class RankNet(nn.Module):
        def __init__(self, qdim, kdim):
            super().__init__()
            self.query = nn.Sequential(nn.Linear(qdim, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 256))
            self.key = nn.Sequential(nn.Linear(kdim, 512), nn.GELU(), nn.Linear(512, 256))
            self.log_scale = nn.Parameter(torch.tensor(float(np.log(20.))))

    class ContextNet(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.query = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 256))
            self.key = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Linear(512, 256))
            self.log_scale = nn.Parameter(torch.tensor(float(np.log(20.))))
            self.grip = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 6))

    class MagnitudeNet(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))

        def forward(self, x):
            return self.net(x)

    return RankNet, ContextNet, MagnitudeNet


def _prefix_feature_fn(model, *, return_tokens=False):
    """Return pooled and optionally per-token final VLM features; no action call."""
    import jax
    import flax.nnx as nnx
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask
    from openpi.shared.nnx_utils import module_jit

    class PrefixFeature(nnx.Module):
        def __init__(self, base):
            self.base = base

        def __call__(self, observation):
            original_batch = observation.state.shape[0]
            if return_tokens and original_batch == 1:
                # The token cache was extracted in batches of four. Matching
                # that prefix batch shape avoids backend-dependent attention
                # rounding differences between batch-1 serving and batch-4
                # extraction; only the first identical copy is returned.
                observation = jax.tree.map(
                    lambda x: jnp.repeat(x, 4, axis=0) if hasattr(x, "ndim") and x.ndim else x,
                    observation,
                )
            obs = model_api.preprocess_observation(None, observation, train=False)
            tokens, valid, ar = self.base.embed_prefix(obs)
            positions = jnp.cumsum(valid, axis=1) - 1
            (hidden, _), _ = self.base.PaliGemma.llm(
                [tokens, None], mask=make_attn_mask(valid, ar), positions=positions
            )
            weights = valid.astype(hidden.dtype)[..., None]
            pooled = jnp.sum(hidden * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1)
            if return_tokens:
                return (pooled[:original_batch].astype(jnp.float16),
                        hidden[:original_batch].astype(jnp.float16), valid[:original_batch])
            return pooled.astype(jnp.float16)

    return module_jit(PrefixFeature(model).__call__)


def append_history(history, mask, actions, horizon=10):
    """Append planned memory actions and return left-zero-padded fork-order history."""
    h = np.asarray(history, np.float32)
    m = np.asarray(mask, bool)
    a = np.asarray(actions, np.float32)
    if h.shape != (horizon, 12) or m.shape != (horizon,) or a.ndim != 2 or a.shape[1] != 12:
        raise ValueError("history/actions have invalid shape")
    valid = h[m]
    joined = np.concatenate((valid, a), axis=0)[-horizon:]
    out = np.zeros((horizon, 12), np.float32)
    out_mask = np.zeros((horizon,), bool)
    if len(joined):
        out[-len(joined):] = joined
        out_mask[-len(joined):] = True
    return out, out_mask


def unit_metadata(pool):
    """Match magnitude_head._unit_metadata: DCT key plus log raw EEF magnitude."""
    motion = np.asarray(pool["action10"], np.float32)[:, :, 5:11]
    magnitude = np.sqrt(np.mean(np.sum(motion.astype(np.float64) ** 2, axis=2), axis=1)).astype(np.float32)
    key = np.concatenate((np.asarray(pool["key"], np.float32), np.log(np.maximum(magnitude, 1e-6))[:, None]), axis=1)
    return key.astype(np.float32), magnitude


def materialize_unit(policy_actions, scale, grip_signs, *, use_context_gripper=True,
                     scale_position_only=False):
    """Scale stored EEF motion and optionally replace its initial grip steps.

    The two switches are ablations; both default to the historical behavior.
    """
    out = np.asarray(policy_actions, np.float32).copy()
    grip = np.asarray(grip_signs, np.float32).reshape(-1)
    if out.ndim != 2 or out.shape[1] != 12 or len(out) == 0 or len(grip) < min(5, len(out)):
        raise ValueError("expected nonempty [T,12] policy actions and at least five grip predictions")
    if scale_position_only:
        out[:, :3] *= float(scale)
    else:
        out[:, :6] *= float(scale)
    if use_context_gripper:
        n = min(5, len(out))
        out[:n, 6] = grip[:n]
    return out


def clip_magnitude_multiplier(base_scale, multiplier, minimum=.25, maximum=4.0):
    """Apply an ablation multiplier, then enforce the policy's scale bounds."""
    if not np.isfinite(base_scale) or not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("magnitude scale and multiplier must be finite; multiplier positive")
    return float(np.clip(float(base_scale) * float(multiplier), minimum, maximum))


class MemoryOnlyPolicy:
    def __init__(self, args):
        import torch
        import jax
        import jax.numpy as jnp
        from openpi import transforms
        from openpi.models import model as model_api
        from openpi.shared import normalize
        from openpi.training import config as train_config

        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
        torch.set_num_interop_threads(1)
        self.args = args
        self.lock = threading.RLock()
        self.histories = {}
        self.log_lock = threading.Lock()
        self.log_path = Path(args.query_log) if args.query_log else None
        self.diagnostic_log_path = Path(args.diagnostic_log) if getattr(args, "diagnostic_log", None) else None
        self.diagnostic_reference_count = int(getattr(args, "diagnostic_reference_count", 0))
        self.diagnostic_count = 0
        self.action_expert_calls = 0

        checkpoint = Path(args.checkpoint)
        self.stats = normalize.load(checkpoint / "assets")
        self.model, _ = expert_api._load_expert_model(checkpoint)
        self.bridge_t0 = float(getattr(args, "bridge_t0", 0.2))
        self.bridge_checkpoint = Path(args.bridge_checkpoint) if getattr(args, "bridge_checkpoint", None) else None
        self.bridge_steps = int(getattr(args, "bridge_steps", 10))
        self.bridge_noise_seed = int(getattr(args, "bridge_noise_seed", 4101))
        self.bridge_fn = None
        self.pretrained_normal_fn = None
        use_base_bridge = bool(getattr(args, "bridge_use_base_expert", False))
        if self.bridge_checkpoint is not None and use_base_bridge:
            raise ValueError("choose either --bridge-checkpoint or --bridge-use-base-expert")
        if self.bridge_checkpoint is not None or use_base_bridge:
            if self.diagnostic_reference_count > 0:
                # Preserve a base-checkpoint policy for a bounded diagnostic.
                # Its normal sample shares the exact observation and noise
                # with the Expert bridge output.
                if self.bridge_checkpoint is None:
                    self.pretrained_model = self.model
                else:
                    self.pretrained_model, _ = expert_api._load_expert_model(checkpoint)
                from openpi.shared.nnx_utils import module_jit
                self.pretrained_normal_fn = module_jit(
                    self.pretrained_model.sample_actions, static_argnames=("num_steps",)
                )
            if self.bridge_checkpoint is not None:
                expert_api._overlay_expert(self.model, self.bridge_checkpoint)
            from openpi.shared.nnx_utils import module_jit
            self.bridge_fn = module_jit(self.model.sample_bridge, static_argnames=("num_steps",))
        elif self.diagnostic_reference_count > 0:
            raise ValueError("--diagnostic-reference-count requires a bridge mode")
        self.prefix_fn = _prefix_feature_fn(self.model, return_tokens=args.latent_query_checkpoint is not None)
        cfg = train_config.get_config("pi05_robocasa_multitask")
        dc = cfg.data.create(cfg.assets_dirs, cfg.model)
        self.input_transform = transforms.compose([
            transforms.InjectDefaultPrompt(None), *dc.data_transforms.inputs,
            transforms.Normalize(self.stats, use_quantiles=False), *dc.model_transforms.inputs,
        ])

        context_dir = Path(args.context_dir)
        key_ck = torch.load(Path(args.key_checkpoint), map_location="cpu", weights_only=False)
        if not key_ck.get("input_history", False):
            raise ValueError("expected history-aware Context checkpoint")
        self.key_ck = key_ck
        RankNet, ContextNet, MagnitudeNet = _network_classes()
        self.latent_net = None
        self.latent_ck = None
        if args.latent_query_checkpoint is None:
            self.rank_net = RankNet(int(key_ck["q_dim"]), int(key_ck["k_dim"]))
            self.rank_net.load_state_dict(key_ck["state"], strict=True)
            self.rank_net.eval()
        else:
            from latent_query import _network_classes as _latent_classes
            self.latent_ck = torch.load(Path(args.latent_query_checkpoint), map_location="cpu", weights_only=False)
            LatentQuery = _latent_classes()
            self.latent_net = LatentQuery(int(self.latent_ck["query_dim"]), int(self.latent_ck["candidate_dim"]),
                                          int(self.latent_ck["token_dim"]), int(self.latent_ck["width"]))
            self.latent_net.load_state_dict(self.latent_ck["state"], strict=True)
            self.latent_net.eval()

        self.context_rows = np.load(context_dir / "cache/rows.npz", allow_pickle=False)
        self.feature_table = np.asarray(np.load(context_dir / "cache/prefix_mean_fp16.npy", mmap_mode="r"), np.float32)
        self.context_pool = dict(np.load(context_dir / "cache/pool.npz", allow_pickle=False))
        self.pool = dict(np.load(Path(args.memory_dir) / "pools/U_memory.npz", allow_pickle=False))
        self._check_memory_alignment()
        feature_mean, feature_std, state_mean, state_std = key_ck["norm"]
        candidate_context = np.asarray(self.context_rows["candidate_context"], np.int64)
        candidate_state = np.asarray(self.context_rows["candidate_state"], np.float32)
        self.key_norm = tuple(np.asarray(x, np.float32) for x in (feature_mean, feature_std, state_mean, state_std))
        fm, fs, sm, ss = self.key_norm
        candidate_input = np.concatenate([
            (self.feature_table[candidate_context] - fm) / fs,
            (candidate_state - sm) / ss,
        ], axis=1).astype(np.float32)
        if len(candidate_input) != len(self.pool["length"]):
            raise AssertionError("candidate key rows do not align with memory U order")
        with torch.no_grad():
            if self.latent_net is None:
                self.candidate_embeddings = torch.nn.functional.normalize(
                    self.rank_net.key(torch.as_tensor(candidate_input, dtype=torch.float32)), dim=-1
                )
            else:
                latent_types = np.asarray(self.pool["shape"], np.float32).reshape(len(self.pool["length"]), -1)
                self.candidate_embeddings = torch.nn.functional.normalize(
                    self.latent_net.key(torch.as_tensor(latent_types, dtype=torch.float32)), dim=-1
                )

        grip_ck = torch.load(Path(args.gripper_checkpoint), map_location="cpu", weights_only=False)
        self.grip_norm = grip_ck["norm"]
        self.grip_net = ContextNet(int(grip_ck["input_dim"]))
        # The Context checkpoint has exactly this network, including the grip head.
        self.grip_net.load_state_dict(grip_ck["state"], strict=True)
        self.grip_net.eval()

        mag_ck = torch.load(Path(args.magnitude_checkpoint), map_location="cpu", weights_only=False)
        self.mag_ck = mag_ck
        self.mag_net = MagnitudeNet(int(mag_ck["input_dim"]))
        self.mag_net.load_state_dict(mag_ck["state"], strict=True)
        self.mag_net.eval()
        self.memory_meta, self.unit_magnitude = unit_metadata(self.pool)
        # History-free magnitude heads consume the normalized Context query
        # (prefix mean + state), whereas the ordinary head consumes the
        # history-augmented rank query.  Keep retrieval history-aware in both.
        q_dim_for_magnitude = candidate_input.shape[1] if self.mag_ck.get("history_free", False) else int(key_ck["q_dim"])
        expected_mag_dim = int(q_dim_for_magnitude) + self.memory_meta.shape[1]
        if self.mag_ck["input_dim"] != expected_mag_dim:
            raise AssertionError("magnitude feature layout mismatch")
        self._extra = (torch, jax, jnp, model_api)

        print(json.dumps({"serving": "rm-memory-bridge" if self.bridge_fn is not None else "rm-memory-only", "port": args.port,
            "memory_units": len(self.pool["length"]), "context_checkpoint": args.key_checkpoint,
            "magnitude_checkpoint": args.magnitude_checkpoint, "gripper_checkpoint": args.gripper_checkpoint,
            "bridge_checkpoint": str(self.bridge_checkpoint) if self.bridge_checkpoint else None,
            "bridge_t0": self.bridge_t0 if self.bridge_fn is not None else None,
            "prefix_only": True, "action_expert_calls": self.action_expert_calls,
            "subtask_generation": os.environ.get("SKILLNET_SUBTASK_GENERATE", "0")}, ensure_ascii=False, default=str), flush=True)

    def _check_memory_alignment(self):
        for name in ("task", "episode", "global_episode", "frame", "length", "action_offsets", "key"):
            if not np.array_equal(np.asarray(self.pool[name]), np.asarray(self.context_pool[name])):
                raise ValueError(f"memory and trained Context candidate pool differ: {name}")
        if not np.array_equal(np.asarray(self.pool["actions_flat"]), np.asarray(self.context_pool["actions_flat"])):
            raise ValueError("memory action rows differ from the Context candidate snapshot")
        for pool_key, row_key in (("task", "candidate_task"), ("global_episode", "candidate_episode"), ("frame", "candidate_frame")):
            if not np.array_equal(np.asarray(self.pool[pool_key]), np.asarray(self.context_rows[row_key])):
                raise ValueError(f"memory and Context candidate order differs: {pool_key}")

    def _log(self, record):
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _log_diagnostic(self, record):
        if self.diagnostic_log_path is None:
            return
        self.diagnostic_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_lock, self.diagnostic_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _context_query(self, prefix, state):
        torch, *_ = self._extra
        fm, fs, sm, ss = self.key_norm
        return np.concatenate(((np.asarray(prefix, np.float32).reshape(-1) - fm) / fs,
                               (np.asarray(state, np.float32).reshape(-1) - sm) / ss)).astype(np.float32)

    def _choose(self, q_context, history, mask, *, prefix_tokens=None, prefix_mask=None, state=None, prefix=None):
        torch, *_ = self._extra
        if self.latent_net is not None:
            norm = self.latent_ck["feature_norm"]
            p = np.asarray(prefix, np.float32).reshape(-1)
            s = np.asarray(state, np.float32).reshape(-1)
            qctx = np.concatenate(((p-np.asarray(norm["prefix_mean"],np.float32))/np.asarray(norm["prefix_std"],np.float32),
                                   (s-np.asarray(norm["state_mean"],np.float32))/np.asarray(norm["state_std"],np.float32))).astype(np.float32)
            hmean=np.asarray(norm["history_mean"],np.float32);hstd=np.asarray(norm["history_std"],np.float32)
            hn=((np.asarray(history,np.float32)-hmean)/hstd)*np.asarray(mask,bool)[:,None]
            q=np.concatenate((qctx,hn.reshape(-1),np.asarray(mask,np.float32))).astype(np.float32)
            with torch.no_grad():
                ts=torch.as_tensor(np.asarray(prefix_tokens,np.float32),dtype=torch.float32)
                tm=torch.as_tensor(np.asarray(prefix_mask,bool),dtype=torch.bool)
                qt=torch.as_tensor(q[None],dtype=torch.float32)
                qemb=self.latent_net.encode_query(ts,tm,qt)
                scores=(qemb @ self.candidate_embeddings.T)[0]*self.latent_net.log_scale.exp()
                selected=int(torch.argmax(scores).item())
            return selected,float(scores[selected].item()),q_context
        hmean = np.asarray(self.key_ck["hist_norm"]["mean"], np.float32)
        hstd = np.asarray(self.key_ck["hist_norm"]["std"], np.float32)
        hist_norm = ((history - hmean) / hstd) * mask[:, None]
        q = np.concatenate((q_context, hist_norm.reshape(-1), mask.astype(np.float32))).astype(np.float32)
        qt = torch.as_tensor(q[None], dtype=torch.float32)
        with torch.no_grad():
            qemb = torch.nn.functional.normalize(self.rank_net.query(qt), dim=-1)
            scores = (qemb @ self.candidate_embeddings.T)[0] * self.rank_net.log_scale.exp()
        selected = int(torch.argmax(scores).item())
        return selected, float(scores[selected].item()), q

    def _gripper(self, q_context):
        torch, *_ = self._extra
        with torch.no_grad():
            logits = self.grip_net.grip(torch.as_tensor(q_context[None], dtype=torch.float32))[0]
        return (logits[:5] >= 0).to(torch.float32).numpy() * 2.0 - 1.0

    def _scale(self, rank_query, selected, q_context):
        torch, *_ = self._extra
        # NH magnitude checkpoints deliberately omit the action-history input;
        # retrieval still uses the normal Context+history ranker.
        magnitude_query = q_context if self.mag_ck.get("history_free", False) else rank_query
        x = np.concatenate((magnitude_query, self.memory_meta[selected])).astype(np.float32)
        xx = (x - np.asarray(self.mag_ck["feature_mean"], np.float32)) / np.asarray(self.mag_ck["feature_std"], np.float32)
        with torch.no_grad():
            log_s = float(self.mag_net(torch.as_tensor(xx[None], dtype=torch.float32))[0, 0].item())
        # Sensitivity conditions scale the learned prediction, then apply the
        # same operational bounds used by the base memory-only policy.
        return clip_magnitude_multiplier(np.exp(log_s), self.args.magnitude_multiplier)

    def infer(self, raw_observation):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_api
        from memory_server_utils import ClientHistory

        obs_in = dict(raw_observation)
        client_id = str(obs_in.pop("client_id", "default"))
        episode_id = str(obs_in.pop("episode_id", "0"))
        episode_start = bool(obs_in.pop("episode_start", False))
        state = np.asarray(obs_in["observation/state"], np.float32)
        if state.shape != (16,):
            raise ValueError(f"expected fork-order state[16], got {state.shape}")
        with self.lock:
            diagnostic_index = None
            if (self.pretrained_normal_fn is not None
                    and self.diagnostic_count < self.diagnostic_reference_count):
                diagnostic_index = self.diagnostic_count
                self.diagnostic_count += 1
            history_obj = self.histories.setdefault(client_id, ClientHistory(10))
            history, mask, request_index = history_obj.begin_request(episode_id, episode_start)
            start = time.perf_counter()
            transformed = self.input_transform(obs_in)
            batched = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], transformed)
            observation = model_api.Observation.from_dict(batched)
            prefix_features = self.prefix_fn(observation)
            if self.latent_net is None:
                prefix = np.asarray(prefix_features, np.float32)[0]
                prefix_tokens = prefix_mask = None
            else:
                prefix, prefix_tokens, prefix_mask = prefix_features
                prefix = np.asarray(prefix, np.float32)[0]
                prefix_tokens = np.asarray(prefix_tokens, np.float16)
                prefix_mask = np.asarray(prefix_mask, bool)
            if prefix.shape != (2048,):
                raise ValueError(f"prefix feature must be [2048], got {prefix.shape}")
            q_context = self._context_query(prefix, state)
            planned = np.zeros((50, 12), np.float32)
            filled = 0
            selected_records = []
            local_history, local_mask = history.copy(), mask.copy()
            while filled < 50:
                selected, score, rank_query = self._choose(q_context, local_history, local_mask,
                    prefix_tokens=prefix_tokens, prefix_mask=prefix_mask, state=state, prefix=prefix)
                off0, off1 = int(self.pool["action_offsets"][selected]), int(self.pool["action_offsets"][selected + 1])
                stored = np.asarray(self.pool["actions_flat"][off0:off1], np.float32)
                unit = expert_api.ex.action_storage_to_policy(stored).astype(np.float32)
                if not len(unit):
                    raise ValueError(f"empty memory unit {selected}")
                scale = self._scale(rank_query, selected, q_context)
                grip_context = np.concatenate(((prefix - self.grip_norm["feature_mean"]) / self.grip_norm["feature_std"],
                                               (state - self.grip_norm["state_mean"]) / self.grip_norm["state_std"])).astype(np.float32)
                grip = self._gripper(grip_context)
                # The learned Context grip head predicts the next five commands;
                # later values in this returned horizon are memory values and are
                # normally discarded by the five-step replanning client.
                use_context_gripper = not self.args.draft_unit_gripper or filled == 0
                unit = materialize_unit(
                    unit, scale, grip,
                    use_context_gripper=use_context_gripper,
                    scale_position_only=self.args.scale_position_only,
                )
                take = min(len(unit), 50 - filled)
                segment = unit[:take].copy()
                planned[filled:filled + take] = segment
                local_history, local_mask = append_history(local_history, local_mask, segment, 10)
                selected_records.append({"unit_id": selected, "task_id": int(self.pool["task"][selected]),
                    "episode": int(self.pool["episode"][selected]), "start": int(self.pool["frame"][selected]),
                    "length": int(self.pool["length"][selected]), "score": score,
                    "magnitude": scale, "gripper_first5": grip.tolist(), "steps_used": int(take)})
                filled += take
            if planned.shape != (50, 12) or not np.isfinite(planned).all():
                raise AssertionError("memory-only output must be finite [50,12]")
            returned = planned
            bridge_seconds = 0.0
            if self.bridge_fn is not None:
                # The retrieval-only proposal is converted to the policy's
                # normalized 32-D action space and refined from bridge_t0.
                # No noise-to-action (ordinary π0.5) sampling is invoked.
                from openpi.shared import normalize
                bridge_start = time.perf_counter()
                raw = np.zeros((1, 50, 32), np.float32)
                raw[0, :, :12] = planned
                action_stats = self.stats["actions"]
                mean = np.asarray(action_stats.mean, np.float32)[:32]
                std = np.asarray(action_stats.std, np.float32)[:32]
                draft_norm = (raw - mean[None, None, :]) / (std[None, None, :] + 1e-6)
                key = jax.random.fold_in(jax.random.key(self.bridge_noise_seed), int(request_index))
                noise = jax.random.normal(jax.random.fold_in(key, 1), (1, 50, 32))
                refined = self.bridge_fn(jax.random.fold_in(key, 2), observation,
                    jnp.asarray(draft_norm), jnp.asarray([self.bridge_t0], jnp.float32),
                    num_steps=self.bridge_steps, noise=noise)
                refined = np.asarray(refined, np.float32)[0] * (std[None, :] + 1e-6) + mean[None, :]
                returned = refined[:, :12].astype(np.float32)
                if self.args.grip_after_bridge:
                    if not selected_records:
                        raise AssertionError("bridge grip override requires the first unit's Context prediction")
                    grip_override = np.asarray(selected_records[0]["gripper_first5"], np.float32)
                    returned[:min(5, len(returned)), 6] = grip_override[:min(5, len(returned))]
                self.action_expert_calls += 1
                bridge_seconds = time.perf_counter() - bridge_start
            history_obj.set_pending(returned[:int(self.args.replan_steps)])
            elapsed = time.perf_counter() - start
            self._log({"record_type": "memory_bridge_query" if self.bridge_fn is not None else "memory_only_query", "client_id": client_id,
                "episode_id": episode_id, "query_index": request_index,
                "history_valid": int(mask.sum()), "selected_units": selected_records,
                "action_expert_calls": self.action_expert_calls,
                "action_expert_calls_bridge": int(self.bridge_fn is not None),
                "action_expert_called": self.bridge_fn is not None, "bridge_seconds": bridge_seconds,
                "bridge_t0": self.bridge_t0 if self.bridge_fn is not None else None,
                "inference_seconds": elapsed,
                "magnitude_multiplier": float(self.args.magnitude_multiplier),
                "draft_unit_gripper": bool(self.args.draft_unit_gripper),
                "grip_after_bridge": bool(self.args.grip_after_bridge),
                "scale_position_only": bool(self.args.scale_position_only),
                "position_step_norm_median": float(np.median(np.linalg.norm(
                    planned[:int(self.args.replan_steps), :3].astype(np.float64), axis=1))),
                "shape": list(planned.shape), "finite": True})
            if diagnostic_index is not None:
                # This is a like-for-like reference: same observation,
                # normalized-action statistics and Gaussian noise tensor.
                normal_norm = self.pretrained_normal_fn(
                    jax.random.fold_in(key, 3), observation,
                    num_steps=self.bridge_steps, noise=noise,
                )
                normal_physical = (
                    np.asarray(normal_norm, np.float32)[0] * (std[None, :] + 1e-6)
                    + mean[None, :]
                )
                draft5 = planned[:5].astype(np.float32)
                expert5 = returned[:5].astype(np.float32)
                normal5 = normal_physical[:5, :12].astype(np.float32)

                def _rms(lhs, rhs, dims):
                    return float(np.sqrt(np.mean((lhs[:, :dims] - rhs[:, :dims]) ** 2)))

                self._log_diagnostic({
                    "diagnostic_index": diagnostic_index,
                    "episode_id": episode_id,
                    "query_index": request_index,
                    "bridge_t0": self.bridge_t0,
                    "bridge_steps": self.bridge_steps,
                    "noise_shared": True,
                    "draft_first5_physical_fork_order": draft5.tolist(),
                    "expert_first5_physical_fork_order": expert5.tolist(),
                    "pretrained_pi05_first5_physical_fork_order": normal5.tolist(),
                    "expert_vs_draft_rms12": _rms(expert5, draft5, 12),
                    "expert_vs_draft_position_rms3": _rms(expert5, draft5, 3),
                    "expert_vs_pi05_rms12": _rms(expert5, normal5, 12),
                    "expert_vs_pi05_position_rms3": _rms(expert5, normal5, 3),
                    "draft_vs_pi05_rms12": _rms(draft5, normal5, 12),
                    "draft_vs_pi05_position_rms3": _rms(draft5, normal5, 3),
                })
            return {"actions": returned}


def main():
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    p = argparse.ArgumentParser()
    state = Path(os.environ.get("SKILLNET_STATE_DIR", "/work/.skillnet-state"))
    root = state / "outputs/robocasa-memory"
    p.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("SKILLNET_ROBOCASA_CHECKPOINT", state / "checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000")))
    p.add_argument("--memory-dir", type=Path, default=root / "memory-01/policy-order-v2")
    p.add_argument("--context-dir", type=Path, default=root / "ctx-01")
    p.add_argument("--key-checkpoint", type=Path, default=root / "key-lite-01/context_history.pt")
    p.add_argument("--latent-query-checkpoint", type=Path, default=None)
    p.add_argument("--magnitude-checkpoint", type=Path, default=root / "key-lite-01/magnitude/magnitude_head.pt")
    p.add_argument("--magnitude-multiplier", type=float, default=1.0)
    p.add_argument("--draft-unit-gripper", action="store_true",
                   help="use Context grip only for the first unit; later units keep stored grip values")
    p.add_argument("--grip-after-bridge", action="store_true",
                   help="replace the returned chunk's first five grip commands with Context predictions")
    p.add_argument("--scale-position-only", action="store_true",
                   help="apply the learned magnitude multiplier only to EEF translation, not rotation")
    p.add_argument("--bridge-checkpoint", type=Path, default=None,
                   help="overlay an action-expert bridge checkpoint on the pretrained model")
    p.add_argument("--bridge-use-base-expert", action="store_true",
                   help="run the bridge directly with the released 75k action expert")
    p.add_argument("--bridge-t0", type=float, default=0.2)
    p.add_argument("--bridge-steps", type=int, default=10)
    p.add_argument("--bridge-noise-seed", type=int, default=4101)
    p.add_argument("--gripper-checkpoint", type=Path, default=root / "ctx-01/context_model.pt")
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--query-log", type=Path)
    p.add_argument("--diagnostic-log", type=Path)
    p.add_argument("--diagnostic-reference-count", type=int, default=0)
    a = p.parse_args()
    if a.magnitude_multiplier <= 0:
        p.error("--magnitude-multiplier must be positive")
    if a.diagnostic_reference_count < 0:
        p.error("--diagnostic-reference-count must be nonnegative")
    os.environ["SKILLNET_SUBTASK_GENERATE"] = "0"
    policy = MemoryOnlyPolicy(a)
    policy_name = "memory-bridge" if policy.bridge_fn is not None else "memory-only"
    print(f"serving rm-{policy_name} on {a.host}:{a.port}; action_expert_calls={policy.action_expert_calls}", flush=True)
    WebsocketPolicyServer(policy, host=a.host, port=a.port, metadata={"policy": policy_name, "action_horizon": 50},
                          ping_interval=None, ping_timeout=None, infer_in_thread=True).serve_forever()


if __name__ == "__main__":
    main()
