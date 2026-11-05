"""Extended saved-state branch policy with four-key retrieval and raw drafts.

The original rescue_branching/serve.py remains the compatibility server for
the previously checked-in 26-condition route. This sibling adds the stage-3
pre-registered key union and fixed-start-noise branch conditions.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent
sys_path = str(PI05)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from key_merge import (KEY_ORDER, action_key_scores, codec_cosine_scores,
                       merge_key_rankings, rank_scores, raw_memory_draft)
from run import load_rows, restore_codec

_BASE_SPEC = importlib.util.spec_from_file_location(
    'rescue_branching_verified_serve', HERE/'serve.py')
_BASE_MODULE = importlib.util.module_from_spec(_BASE_SPEC)
assert _BASE_SPEC and _BASE_SPEC.loader
_BASE_SPEC.loader.exec_module(_BASE_MODULE)
BranchPolicy = _BASE_MODULE.BranchPolicy


def wrist_group_scores(index, query, query_groups, *, block=None):
    """Exact group-1 MaxSim, using make_vlm_keys' wrist-only definition."""
    torch = index.torch
    block = index.block if block is None else int(block)
    qgroups = np.asarray(query_groups)
    q = torch.as_tensor(np.asarray(query).copy(), device=index.device).float()
    q = (q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-8)).half()
    selected = q[qgroups == 1]
    if not selected.shape[0]:
        raise ValueError('query has no wrist-camera tokens')
    left, right = index.ranges[1]
    scores = torch.empty(len(index.ids), device=index.device, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, len(index.ids), block):
            end = min(start + block, len(index.ids))
            sim = torch.matmul(selected[None], index.keys[start:end, left:right].transpose(1, 2)).float()
            sim.masked_fill_(index.groups[start:end, None, left:right] != 1, -torch.inf)
            scores[start:end] = sim.amax(-1).mean(-1)
    return scores


class ExtendedBranchPolicy:
    def __init__(self, run: Path, checkpoint: Path):
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        # The inherited policy and token index are the exact already-verified
        # path; do not change its normal/current behavior.
        self.wrapper = BranchPolicy(run, checkpoint)
        self.base = self.wrapper.base  # existing verified MemoryPolicy instance
        self.run = run
        self.rows = load_rows(run)
        self.memory_ids = np.flatnonzero(self.rows['memory']).astype(np.int64)
        if not np.array_equal(self.memory_ids, np.asarray(self.base.index.ids, np.int64)):
            raise AssertionError('VLM index IDs differ from run-01 memory row IDs')
        self.codec = restore_codec(run)
        pooled = np.load(run / 'pooled.npy', mmap_mode='r', allow_pickle=False)
        mem_b, _ = self.codec.prior(np.asarray(pooled[self.memory_ids], np.float32),
                                    np.asarray(self.rows['state'][self.memory_ids], np.float32))
        self.mem_b = np.asarray(mem_b, np.float32)
        _, self.b_mean, self.b_std = codec_cosine_scores(self.mem_b[:1], self.mem_b)
        self.mem_b_unit = (self.mem_b - self.b_mean) / self.b_std
        self.mem_b_unit /= np.maximum(np.linalg.norm(self.mem_b_unit, axis=1, keepdims=True), 1e-8)
        self.mem_actions = self.rows['actions'][self.memory_ids]

    def _obs(self, request):
        from openpi.models.model import Observation
        raw = {k: request[k] for k in ['observation/image', 'observation/wrist_image',
                                       'observation/state', 'prompt']}
        row = self.base.transform(raw)
        obs = Observation(
            images={k:self.base.jnp.asarray(v)[None].astype(self.base.jnp.float32)/255*2-1
                    for k,v in row['image'].items()},
            image_masks={k:self.base.jnp.asarray([v]) for k,v in row['image_mask'].items()},
            state=self.base.jnp.asarray(row['state'])[None],
            tokenized_prompt=self.base.jnp.asarray(row['tokenized_prompt'])[None],
            tokenized_prompt_mask=self.base.jnp.asarray(row['tokenized_prompt_mask'])[None])
        return row, obs

    def _rankings(self, hidden, groups, state, normal_action):
        index = self.base.index
        ids = self.memory_ids
        vlm = index.scores(hidden, groups).float().cpu().numpy()
        wrist = wrist_group_scores(index, hidden, groups).float().cpu().numpy()
        pooled_feature = np.asarray(hidden)[np.asarray(groups) >= 0].mean(0, keepdims=True).astype(np.float32)
        b_hat, _ = self.codec.prior(pooled_feature, np.asarray(state, np.float32))
        b_hat = np.asarray(b_hat, np.float32).reshape(1, -1)
        b_scores, _, _ = codec_cosine_scores(b_hat, self.mem_b, self.b_mean, self.b_std)
        predicted = np.asarray(normal_action, np.float32)[0, :, :7]
        action_scores = action_key_scores(predicted, self.mem_actions)
        score_map = {
            'vlm_current': vlm,
            'wrist_only': wrist,
            'codec_prior_b': b_scores,
            'pi05_pred_to_memory_action': action_scores,
        }
        rankings = {name: rank_scores(score_map[name], ids) for name in KEY_ORDER}
        merged = merge_key_rankings(rankings, per_key=2, target=8)
        return rankings, merged, pooled_feature, b_hat

    def infer(self, request):
        started = time.monotonic()
        row, obs = self._obs(request)
        key = self.wrapper.jax.random.key(int(request['s2_rng_seed']))
        condition = request.get('branch_condition', 'normal_rank')
        debug = {'condition': condition, 'rng_seed': int(request['s2_rng_seed'])}
        if condition in ('normal', 'normal_alt', 'normal_rank'):
            result = self.base.normal(key, obs, num_steps=10)
            if condition == 'normal_rank':
                hidden, groups = self.base.features(obs)
                hidden, groups = np.asarray(hidden)[0], np.asarray(groups)[0]
                rankings, merged, feature, b_hat = self._rankings(
                    hidden, groups, np.asarray(obs.state), result)
                current_top = self.base.index.search_topk(hidden, groups, 8)
                debug['top8'] = [{'memory_row': r, 'score': s} for r, s in current_top]
                debug['four_key_top8'] = {
                    name: rankings[name][:8] for name in KEY_ORDER
                }
                debug['merged_candidates'] = merged
                debug['query_b_hat_norm'] = float(np.linalg.norm(b_hat))
                debug['vlm_pooled_feature_norm'] = float(np.linalg.norm(feature))
            if condition == 'normal_alt':
                debug['normal_rng_changed'] = True
        elif condition == 'zero_fixed':
            t0 = float(request['branch_t0'])
            draft = np.zeros((1, 10, 32), np.float32)
            result = self.base.bridge(key, obs, self.wrapper.jnp.asarray(draft),
                self.wrapper.jnp.asarray([t0]), num_steps=10)
            debug.update(t0=t0, draft_format='zero')
        elif condition in ('current_fixed', 'candidate_codec', 'candidate_raw'):
            hidden, groups = self.base.features(obs)
            hidden, groups = np.asarray(hidden)[0], np.asarray(groups)[0]
            feature = hidden[groups >= 0].mean(0, keepdims=True).astype(np.float32)
            state = np.asarray(obs.state, np.float32)
            t0 = float(request['branch_t0'])
            if condition == 'current_fixed':
                memory_row, score = self.base.index.search(hidden, groups)
                _, residual = self.base.codec.prior(feature, state)
                draft = self.base.codec.decode(np.asarray(self.base.bases[memory_row])[None],
                                               residual, feature, state)
                debug.update(memory_row=int(memory_row), vlm_score=float(score),
                             t0=t0, draft_format='current_codec')
            elif condition == 'candidate_codec':
                memory_row = int(request['memory_row'])
                _, residual = self.codec.prior(feature, state)
                draft = self.codec.decode(np.asarray(self.base.bases[memory_row])[None],
                    np.asarray(residual), feature, state)
                debug.update(memory_row=memory_row, t0=t0, draft_format='codec_decode',
                             candidate_sources=request.get('candidate_sources', {}))
            else:
                memory_row = int(request['memory_row'])
                # Assert the exact D-raw definition from memory_gap/run.py:
                # rows['actions'][candidate], all normalized 32 dimensions.
                raw = raw_memory_draft(self.rows['actions'], memory_row)
                expected_raw = np.asarray(self.rows['actions'][memory_row], np.float32)
                if not np.array_equal(raw, expected_raw):
                    raise AssertionError('D-raw draft differs from memory_gap rows[actions][candidate]')
                draft = raw[None].copy()
                if draft.shape != (1, 10, 32):
                    raise AssertionError(f'raw action draft has shape {draft.shape}')
                debug.update(memory_row=memory_row, t0=t0, draft_format='D-raw_rows_actions',
                             candidate_sources=request.get('candidate_sources', {}))
            result = self.base.bridge(key, obs, self.wrapper.jnp.asarray(draft),
                self.wrapper.jnp.asarray([t0]), num_steps=10)
            debug['draft_final_mse_normalized'] = float(np.mean(
                (np.asarray(result)[:, :, :7] - np.asarray(draft)[:, :, :7]) ** 2))
        else:
            raise ValueError(f'unknown expanded branch condition {condition!r}')

        actions = np.asarray(result)[0]
        actions = self.base.unnormalize({'actions': actions, 'state': row['state']})['actions'][:, :7]
        if not np.isfinite(actions).all():
            raise FloatingPointError('non-finite action')
        debug['seconds'] = time.monotonic() - started
        return {'actions': actions.astype(np.float32), 's2_debug': debug}


def main():
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    policy = ExtendedBranchPolicy(args.run, args.checkpoint)
    WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port,
        metadata={'experiment':'pi05_memory40_rescue_branching_expanded',
                  'candidate_keys':list(KEY_ORDER),'action_horizon':10,'num_steps':10}).serve_forever()


if __name__ == '__main__':
    main()
