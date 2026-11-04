"""Fixed-score adaptive-noise server for the paired LIBERO-Long experiment.

The base VLM MaxSim retrieval, codec draft, frozen best bridge expert and
pre/postprocessing are reused from pi05_memory/serve.py. Only t0 selection and
the optional low-score normal-policy fallback differ.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
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
SCORE_EDGES = np.asarray([0.92968, 0.94267, 0.95002], np.float64)
MODES = {'adaptive': [1, 1, 2, 3], 'adaptive_fb': [0, 1, 2, 3]}
MODE_T0 = {1: 0.6, 2: 0.4, 3: 0.2}


def observation_seed(observation: dict) -> int:
    """Use the caller's explicit trial/query seed; no server-global RNG state."""
    return int(observation['s2_rng_seed'])

sys.path.insert(0, str(PI05))


def score_mode(score: float, condition: str) -> int:
    if condition not in MODES:
        raise ValueError(f'unknown adaptive condition: {condition}')
    bin_id = int(np.searchsorted(SCORE_EDGES, float(score), side='right'))
    return int(MODES[condition][bin_id])


def append_locked(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(payload, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def normalize_token_keys_blockwise(keys, torch, block_rows=16):
    """Normalize fp16 token keys with fp32 norms without a full-bank temp copy."""
    for start in range(0, int(keys.shape[0]), block_rows):
        end = min(start + block_rows, int(keys.shape[0]))
        denom = torch.linalg.vector_norm(keys[start:end].float(), dim=-1,
                                         keepdim=True).clamp_min(1e-8).half()
        keys[start:end] /= denom
    return keys


class BoundedTokenIndex:
    """Exact base TokenIndex search with bounded normalization temporaries."""
    def __init__(self, output, *, block=32, norm_block=16):
        import torch
        self.torch = torch
        self.ids = np.load(output/'memory_row_ids.npy')
        tokens = np.load(output/'tokens.npy', mmap_mode='r')
        groups = np.load(output/'groups.npy', mmap_mode='r')
        # The base path's whole-bank .float() norm creates a ~36 GiB temporary.
        # Normalize identical fp16 rows in small chunks to stay within our cap.
        self.keys = torch.empty((len(self.ids), 712, 2048), device='cuda', dtype=torch.float16)
        for start in range(0, len(self.ids), norm_block):
            end = min(start + norm_block, len(self.ids))
            chunk = torch.as_tensor(np.asarray(tokens[self.ids[start:end]]),
                                    device='cuda', dtype=torch.float16)
            self.keys[start:end] = normalize_token_keys_blockwise(chunk, torch, norm_block)
            del chunk
        self.groups = torch.as_tensor(np.asarray(groups[self.ids]), device='cuda')
        text = np.flatnonzero(np.any(np.asarray(groups[self.ids])[:,512:] == 2, axis=0))
        self.ranges = [(0,256),(256,512),(512,512+int(text[-1])+1)]
        self.block = block

    def search(self, query, groups):
        torch = self.torch
        with torch.inference_mode():
            q = torch.as_tensor(np.asarray(query).copy(), device='cuda').float()
            q = (q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-8)).half()
            scores = torch.empty(len(self.ids), device='cuda')
            for start in range(0, len(self.ids), self.block):
                end = min(start + self.block, len(self.ids))
                terms = []
                for group, (left, right) in enumerate(self.ranges):
                    selected = q[np.asarray(groups) == group]
                    if not len(selected):
                        raise ValueError('query is missing a required modality')
                    sim = torch.matmul(selected[None],
                        self.keys[start:end,left:right].transpose(1,2)).float()
                    sim.masked_fill_(self.groups[start:end,None,left:right] != group, -torch.inf)
                    terms.append(sim.amax(-1).mean(-1))
                scores[start:end] = torch.stack(terms).mean(0)
            best = int(scores.argmax().item())
            return int(self.ids[best]), float(scores[best])


class AdaptivePolicy:
    def __init__(self, run: Path, checkpoint: Path, status_dir: Path, condition: str,
                 *, verify=True):
        if condition not in MODES:
            raise ValueError(condition)
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.14')
        import torch
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(
                float(os.environ.get('ADAPTIVE_TORCH_MEM_FRACTION', '.20')), 0)
        import flax.nnx as nnx  # noqa: F401 (loads the shared JAX model implementation)
        import jax
        import jax.numpy as jnp

        self.jax, self.jnp = jax, jnp
        self.run, self.output, self.status_dir = run, run, status_dir
        self.condition = condition
        base_path = PI05 / 'serve.py'
        spec = importlib.util.spec_from_file_location('pi05_memory_base_serve_adaptive', base_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'cannot import shared policy server: {base_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Retain exact retrieval math but avoid the base initializer's full-bank
        # fp32 normalization temporary, which exceeds the per-server GPU cap.
        module.TokenIndex = BoundedTokenIndex
        self.base = module.MemoryPolicy(run, checkpoint, normal_only=False)
        self.base.schedule = {'edges': SCORE_EDGES.tolist(), 'modes': MODES[condition],
                              'mode_meaning': ['fallback', 'bridge_t0_0.6',
                                               'bridge_t0_0.4', 'bridge_t0_0.2']}
        self._cache_check = None
        if verify:
            self.verify_fixed_score_table()

    def verify_fixed_score_table(self):
        expected = {
            'adaptive': [score_mode(x, 'adaptive') for x in
                         [0.929679, 0.92968, 0.94267, 0.95002]],
            'adaptive_fb': [score_mode(x, 'adaptive_fb') for x in
                            [0.929679, 0.92968, 0.94267, 0.95002]],
        }
        assert expected == {'adaptive': [1, 1, 2, 3], 'adaptive_fb': [0, 1, 2, 3]}
        self._cache_check = {'score_edges_fixed': SCORE_EDGES.tolist(),
            'adaptive_modes': MODES['adaptive'], 'adaptive_fb_modes': MODES['adaptive_fb'],
            'boundary_behavior': 'searchsorted(side=right); exact edge enters upper bin',
            'token_index_rows': int(len(self.base.index.ids)),
            'bridge_steps': 10, 'normal_fallback_available': True,
            'rng': 'jax.random.key(request.s2_rng_seed); no server-global random stream',
            'allocator_caps': {'XLA_PYTHON_CLIENT_MEM_FRACTION': os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),
                               'torch_per_process_fraction': os.environ.get('ADAPTIVE_TORCH_MEM_FRACTION', '.20')}}

    def infer(self, observation):
        from openpi.models.model import Observation
        started = time.monotonic()
        raw = {key: observation[key] for key in
               ['observation/image', 'observation/wrist_image', 'observation/state', 'prompt']}
        t = time.monotonic()
        row = self.base.transform(raw)
        preprocess_seconds = time.monotonic() - t
        obs = Observation(
            images={k: self.jnp.asarray(v)[None].astype(self.jnp.float32)/255*2-1
                    for k, v in row['image'].items()},
            image_masks={k: self.jnp.asarray([v]) for k, v in row['image_mask'].items()},
            state=self.jnp.asarray(row['state'])[None],
            tokenized_prompt=self.jnp.asarray(row['tokenized_prompt'])[None],
            tokenized_prompt_mask=self.jnp.asarray(row['tokenized_prompt_mask'])[None])
        seed = observation_seed(observation)
        key = self.jax.random.key(seed)

        t = time.monotonic()
        hidden, groups = self.base.features(obs)
        hidden, groups = np.asarray(hidden)[0], np.asarray(groups)[0]
        vlm_seconds = time.monotonic() - t
        t = time.monotonic()
        memory_row, score = self.base.index.search(hidden, groups)
        search_seconds = time.monotonic() - t
        mode = score_mode(score, self.condition)
        fallback = mode == 0
        t0 = MODE_T0.get(mode)

        draft = None
        draft_seconds = 0.0
        if fallback:
            t = time.monotonic()
            result = self.base.normal(key, obs, num_steps=10)
            denoise_seconds = time.monotonic() - t
        else:
            t = time.monotonic()
            feature = hidden[groups >= 0].mean(0)[None]
            state = np.asarray(obs.state)
            _, residual = self.base.codec.prior(feature, state)
            draft = self.base.codec.decode(np.asarray(self.base.bases[memory_row])[None],
                                           residual, feature, state)
            draft_seconds = time.monotonic() - t
            t = time.monotonic()
            result = self.base.bridge(key, obs, self.jnp.asarray(draft),
                                      self.jnp.asarray([t0]), num_steps=10)
            denoise_seconds = time.monotonic() - t

        normalized = np.asarray(result)[0]
        if draft is None:
            delta_l2 = None
            delta_mse = None
        else:
            diff = np.asarray(result)[:, :, :7] - np.asarray(draft)[:, :, :7]
            delta_l2 = float(np.linalg.norm(diff))
            delta_mse = float(np.mean(diff ** 2))
        t = time.monotonic()
        actions = self.base.unnormalize({'actions': normalized, 'state': row['state']})['actions'][:, :7]
        output_transform_seconds = time.monotonic() - t
        actions = actions.astype(np.float32)
        if not np.isfinite(actions).all():
            raise FloatingPointError('nonfinite policy action')
        total_seconds = time.monotonic() - started
        action_hash = hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()
        query_id = int(observation.get('s2_query_id', -1))
        episode_id = str(observation.get('s2_episode_id', 'unknown'))
        trace = {'condition': self.condition, 'episode_id': episode_id,
            'query_id': query_id, 'rng_seed': seed,
            'score': float(score), 'mode': mode, 't0': t0, 'fallback': fallback,
            'memory_row': int(memory_row), 'action_sha256': action_hash,
            'action_chunk': actions.tolist(), 'preprocess_seconds': preprocess_seconds,
            'vlm_seconds': vlm_seconds, 'search_seconds': search_seconds,
            'draft_seconds': draft_seconds, 'denoise_seconds': denoise_seconds,
            'output_transform_seconds': output_transform_seconds,
            'total_seconds': total_seconds}
        append_locked(self.status_dir / 'action_trace.jsonl', trace)
        debug = {'condition': self.condition, 'score': float(score), 'mode': mode,
            't0': t0, 'fallback': fallback, 'memory_row': int(memory_row),
            'num_steps': 10, 'action_sha256': action_hash,
            'draft_final_l2_normalized': delta_l2,
            'draft_final_mse_normalized': delta_mse,
            'preprocess_seconds': preprocess_seconds, 'vlm_seconds': vlm_seconds,
            'search_seconds': search_seconds, 'draft_seconds': draft_seconds,
            'denoise_seconds': denoise_seconds,
            'output_transform_seconds': output_transform_seconds,
            'total_seconds': total_seconds}
        return {'actions': actions, 's2_debug': debug}


def main():
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.14')
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=RUN_DEFAULT)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument('--status-dir', type=Path, required=True)
    parser.add_argument('--condition', choices=sorted(MODES), required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    policy = AdaptivePolicy(args.run, args.checkpoint, args.status_dir, args.condition)
    args.status_dir.mkdir(parents=True, exist_ok=True)
    (args.status_dir / 'server_checks.json').write_text(json.dumps(policy._cache_check, indent=2)+'\n')
    (args.status_dir / 'server_ready.json').write_text(json.dumps({
        'condition': args.condition, 'score_edges': SCORE_EDGES.tolist(),
        'modes': MODES[args.condition], 'mode_meaning': ['fallback','bridge_t0_0.6','bridge_t0_0.4','bridge_t0_0.2'],
        'memory_rows': int(len(policy.base.index.ids)),
        'token_index_memory': 'VLM token MaxSim Top-1', 'nfe': 10}, indent=2)+'\n')
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port,
        metadata={'experiment':'pi05_memory40_adaptive_noise', 'condition':args.condition,
                  'action_horizon':10, 'num_steps':10}).serve_forever()


if __name__ == '__main__':
    main()
