"""LIBERO-Long policy server for codec-prior b drafts.

This intentionally does not construct or load the 19 GB TokenIndex.  It reuses
the frozen codec, dense model and bridge checkpoint from pi05_memory.run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent
RUN_DEFAULT = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
CHECKPOINT_DEFAULT = Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')

sys.path.insert(0, str(PI05))
from run import load_rows, obs_batch, load_policy, restore_codec  # noqa: E402


def standardize_b(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(values, np.float32) - mean) / std


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def retrieve_b(query_b: np.ndarray, memory_b: np.ndarray, memory_ids: np.ndarray, *,
               mean: np.ndarray | None = None, std: np.ndarray | None = None,
               unit_bank: np.ndarray | None = None):
    """Memory-stat standardized cosine search, stable first-row tie break."""
    memory_b = np.asarray(memory_b, np.float32)
    mean = memory_b.mean(0) if mean is None else np.asarray(mean, np.float32)
    std = memory_b.std(0) if std is None else np.asarray(std, np.float32)
    std = np.where(std < 1e-5, 1., std).astype(np.float32)
    bank = unit_rows(standardize_b(memory_b, mean, std)) if unit_bank is None else np.asarray(unit_bank, np.float32)
    q = unit_rows(standardize_b(np.asarray(query_b, np.float32).reshape(1, -1), mean, std))[0]
    scores = bank @ q
    best = int(np.argmax(scores))
    return int(np.asarray(memory_ids)[best]), float(scores[best]), best


class BPriorPolicy:
    def __init__(self, output: Path, checkpoint: Path, *, run: Path = RUN_DEFAULT,
                 verify: bool = True):
        # Set conservative JAX allocator limits before importing JAX.
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.30')
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        import orbax.checkpoint as ocp
        from openpi.shared.nnx_utils import module_jit, PathRegex
        # Reuse the official experiment server's transform, dense-model,
        # feature and unnormalization path without constructing its TokenIndex.
        base_spec = importlib.util.spec_from_file_location('pi05_memory_base_serve', PI05/'serve.py')
        base_module = importlib.util.module_from_spec(base_spec)
        assert base_spec and base_spec.loader
        base_spec.loader.exec_module(base_module)
        ExistingMemoryPolicy = base_module.MemoryPolicy

        self.jax, self.jnp = jax, jnp
        self.output, self.run = output, run
        self.base_policy = ExistingMemoryPolicy(output, checkpoint, normal_only=True)
        self.model = self.base_policy.model
        self.normal = self.base_policy.normal
        self.features = self.base_policy.features
        self.transform = self.base_policy.transform
        self.unnormalize = self.base_policy.unnormalize
        filt = nnx.All(nnx.Param, PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
        selected = output / 'best_bridge.json'
        if not selected.exists():
            selected = output / 'latest_bridge.json'
        partial_path = Path(json.loads(selected.read_text())['path'])
        with ocp.PyTreeCheckpointer() as loader:
            partial = loader.restore(partial_path)
        state_tree = nnx.state(self.model, filt)
        state_tree.replace_by_pure_dict(jax.tree.map(jnp.asarray, partial))
        nnx.update(self.model, state_tree)
        self.bridge = module_jit(self.model.sample_bridge, static_argnames=('num_steps',))

        self.codec = restore_codec(output)  # CPU MLPs; prior inference is intentionally small.
        rows = load_rows(run)
        memory_ids = np.flatnonzero(rows['memory'])
        if len(memory_ids) != 6583:
            raise ValueError(f'expected 6,583 memory rows, got {len(memory_ids)}')
        pooled = np.load(run / 'pooled.npy', mmap_mode='r', allow_pickle=False)
        # This is exactly score_audit.keys.generate_codec's prior definition.
        started = time.monotonic()
        self.memory_b, _ = self.codec.prior(np.asarray(pooled[memory_ids]),
                                            np.asarray(rows['state'][memory_ids]))
        self.memory_b = np.asarray(self.memory_b, np.float32)
        self.b_mean = self.memory_b.mean(0)
        self.b_std = self.memory_b.std(0)
        self.b_std = np.where(self.b_std < 1e-5, 1., self.b_std).astype(np.float32)
        self.memory_b_unit = unit_rows(standardize_b(self.memory_b, self.b_mean, self.b_std))
        self.memory_ids = memory_ids.astype(np.int64)
        self.memory_prior_seconds = time.monotonic() - started

        self.rows, self.pooled = rows, pooled
        if verify:
            self.verify_cached_row()

    def verify_cached_row(self):
        """Assert live-prefix pooling and b-prior agree with the prepared cache."""
        candidates = np.flatnonzero(self.rows['validation'])
        row_id = int(candidates[0])
        obs = obs_batch(self.rows, np.asarray([row_id], np.int64))
        hidden, groups = self.features(obs)
        live_feature = np.asarray(hidden)[0][np.asarray(groups)[0] >= 0].mean(0)
        cached_feature = np.asarray(self.pooled[row_id], np.float32)
        # Prefix encoding is float32, but the cached images went through PIL
        # and the live path through the policy transform.  Allow the observed
        # sub-1e-3 conversion/rounding residual while still checking the actual
        # pooled vector (not merely its dimensionality).
        np.testing.assert_allclose(live_feature, cached_feature, rtol=1e-3, atol=1e-3,
                                   err_msg='server prefix feature differs from pooled.npy')

        cached_state = np.asarray(self.rows['state'][row_id:row_id+1])
        live_b, live_r = self.codec.prior(live_feature[None], cached_state)
        audit_codec = restore_codec(self.output)
        # score_audit.keys.generate_codec calls this exact frozen prior on
        # pooled.npy and rows/state.npy; independently restore it and assert
        # the cached-row b/r values match bit-for-bit within float32 tolerance.
        cached_b, cached_r = self.codec.prior(cached_feature[None], cached_state)
        audit_b, audit_r = audit_codec.prior(cached_feature[None], cached_state)
        live_audit_b, live_audit_r = audit_codec.prior(live_feature[None], cached_state)
        np.testing.assert_allclose(cached_b, audit_b, rtol=0, atol=1e-7,
                                   err_msg='b_hat differs from score_audit codec prior')
        np.testing.assert_allclose(cached_r, audit_r, rtol=0, atol=1e-7,
                                   err_msg='r_hat differs from score_audit codec prior')
        np.testing.assert_allclose(live_b, live_audit_b, rtol=0, atol=1e-7,
                                   err_msg='server b_hat differs from score_audit prior on live feature')
        np.testing.assert_allclose(live_r, live_audit_r, rtol=0, atol=1e-7,
                                   err_msg='server r_hat differs from score_audit prior on live feature')
        self.cache_check = {'row_id': row_id,
            'feature_max_abs_diff': float(np.max(np.abs(live_feature-cached_feature))),
            'cached_b_hat_max_abs_diff_vs_score_audit': float(np.max(np.abs(cached_b-audit_b))),
            'cached_r_hat_max_abs_diff_vs_score_audit': float(np.max(np.abs(cached_r-audit_r))),
            'live_b_hat_max_abs_diff_vs_score_audit_same_feature': float(np.max(np.abs(live_b-live_audit_b))),
            'live_r_hat_max_abs_diff_vs_score_audit_same_feature': float(np.max(np.abs(live_r-live_audit_r))),
            'live_b_hat_max_abs_diff_vs_cached_input': float(np.max(np.abs(live_b-cached_b))),
            'live_r_hat_max_abs_diff_vs_cached_input': float(np.max(np.abs(live_r-cached_r))),
            'score_audit_definition': 'restore_codec(output).prior(pooled[row], state[row])'}

    def infer(self, observation):
        from openpi.models.model import Observation
        started = time.monotonic()
        raw = {key: observation[key] for key in
               ['observation/image', 'observation/wrist_image', 'observation/state', 'prompt']}
        row = self.transform(raw)
        obs = Observation(
            images={k: self.jnp.asarray(v)[None].astype(self.jnp.float32)/255*2-1
                    for k, v in row['image'].items()},
            image_masks={k: self.jnp.asarray([v]) for k, v in row['image_mask'].items()},
            state=self.jnp.asarray(row['state'])[None],
            tokenized_prompt=self.jnp.asarray(row['tokenized_prompt'])[None],
            tokenized_prompt_mask=self.jnp.asarray(row['tokenized_prompt_mask'])[None])
        key = self.jax.random.key(int(observation['s2_rng_seed']))
        condition = observation['s2_condition']
        debug = {'condition': condition, 'fallback': False, 't0': .6, 'num_steps': 10}
        hidden = groups = feature = state = None
        if condition == 'bprior_direct' or condition == 'bprior_retrieve':
            b_started = time.monotonic()
            feature_started = time.monotonic()
            hidden, groups = self.features(obs)
            hidden, groups = np.asarray(hidden)[0], np.asarray(groups)[0]
            feature = hidden[groups >= 0].mean(0)[None].astype(np.float32)
            debug['vlm_feature_seconds'] = time.monotonic() - feature_started
            state = np.asarray(obs.state, np.float32)
            t = time.monotonic()
            query_b, query_r = self.codec.prior(feature, state)
            debug['prior_seconds'] = time.monotonic() - t
            debug['b_hat_seconds'] = debug['prior_seconds']
            debug['b_hat_total_seconds'] = time.monotonic() - b_started
            query_b = np.asarray(query_b, np.float32)
            query_r = np.asarray(query_r, np.float32)
            if condition == 'bprior_direct':
                draft = self.codec.decode(query_b, query_r, feature, state)
                debug['memory_row'] = None
                debug['cosine'] = None
            else:
                retrieve_started = time.monotonic()
                memory_row, cosine, local = retrieve_b(query_b, self.memory_b, self.memory_ids,
                    mean=self.b_mean, std=self.b_std, unit_bank=self.memory_b_unit)
                debug['retrieve_seconds'] = time.monotonic() - retrieve_started
                t = time.monotonic()
                draft = self.codec.decode(self.memory_b[local:local+1], query_r, feature, state)
                debug.update(memory_row=memory_row, cosine=cosine,
                             decode_seconds=time.monotonic()-t)
            result = self._bridge(key, obs, draft, .6)
            debug['draft_final_l2_normalized'] = float(np.linalg.norm(
                np.asarray(result)[:,:,:7] - np.asarray(draft)[:,:,:7]))
            debug['draft_final_mse_normalized'] = float(np.mean(
                (np.asarray(result)[:,:,:7] - np.asarray(draft)[:,:,:7]) ** 2))
        else:
            raise ValueError(f'unsupported bprior condition: {condition}')
        # Return decoder-space difference separately from the executable action
        # transform; both values above are in the same normalized action space.
        actions = np.asarray(result)[0]
        actions = self.unnormalize({'actions': actions, 'state': row['state']})['actions'][:, :7]
        if not np.isfinite(actions).all():
            raise FloatingPointError('nonfinite policy action')
        debug['total_seconds'] = time.monotonic() - started
        return {'actions': actions.astype(np.float32), 's2_debug': debug}

    def _bridge(self, key, obs, draft, t0):
        return self.bridge(key, obs, self.jnp.asarray(draft), self.jnp.asarray([t0]), num_steps=10)


def main():
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.30')
    parser = argparse.ArgumentParser()
    parser.add_argument('--status-dir', type=Path, required=True)
    parser.add_argument('--run', type=Path, default=RUN_DEFAULT)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument('--port', type=int, default=8167)
    args = parser.parse_args()
    policy = BPriorPolicy(args.run, args.checkpoint, run=args.run)
    args.status_dir.mkdir(parents=True, exist_ok=True)
    (args.status_dir / 'server_checks.json').write_text(json.dumps(policy.cache_check, indent=2)+'\n')
    (args.status_dir / 'server_ready.json').write_text(json.dumps({
        'conditions': ['bprior_direct', 'bprior_retrieve'], 't0': .6, 'num_steps': 10,
        'memory_rows': int(len(policy.memory_ids)),
        'memory_prior_seconds': policy.memory_prior_seconds,
        'memory_prior_device': 'cpu', 'token_index_loaded': False,
        'cache_check': policy.cache_check}, indent=2)+'\n')
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port,
        metadata={'experiment':'pi05_bprior_long',
                  'conditions':['bprior_direct','bprior_retrieve'], 'action_horizon':10}).serve_forever()


if __name__ == '__main__':
    main()
