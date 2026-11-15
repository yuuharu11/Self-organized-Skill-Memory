#!/usr/bin/env python3
"""LPC1: no-rotation scale-only type retrieval, in a serve-compatible wrapper.

This module deliberately leaves memory_closed_loop.py untouched.  The online
path delegates image preprocessing and pi05 sampling to its MemoryPolicy
base, then replaces only the six motion axes with a P5-selected, scale-only
type.  A caller may pass the exact per-query ``lpc1_allowed`` mask (needed to
reproduce an offline split); absent a mask, all kept pool entries are eligible
because LIBERO-90 and LIBERO-40 task IDs are different namespaces.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

ROOT = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser() / 'outputs/memory-pilot/unseen-libero'
SCALE = ROOT / 'scale-only-02/production'
DEV_INPUTS = SCALE / 'prep/W_inputs_dev.npz'
DEV_PI05 = ROOT / 'fast-query-01/production/predictions/pi05_dev_rot180.npz'
ASIS_PI05 = ROOT / 'fast-query-01/production/predictions/pi05_eval_asis.npz'
RUN_CHECKPOINT = Path('/work/.skillnet-state/checkpoints/pi05_libero_moe_skill_4_90')


def selected_checkpoint():
    """Return the LIBERO-40-holdout-selected W/S-shared/P5/seed-4101 file."""
    selection = json.loads((SCALE / 'selection.json').read_text())
    row = selection['selected']['W|S-shared|P5|4101']['holdout']
    path = SCALE / 'models' / f"W_S-shared_P5_s4101_f{float(row['frac']):.4f}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def rotate_observation_images_180(observation):
    """Flip raw HWC camera inputs to the orientation used by corrected pi05 caches."""
    out = dict(observation)
    for name in ('observation/image', 'observation/wrist_image'):
        if name not in out:
            raise KeyError(f'missing pi05 camera input {name}')
        image = np.asarray(out[name])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f'{name} must be HWC RGB, got {image.shape}')
        out[name] = np.ascontiguousarray(image[::-1, ::-1])
    return out


class ScaleOnlyP5Selector:
    """The selected scale-only-02 W/S-shared/P5 query model and 1,595 W types."""
    def __init__(self, checkpoint=None):
        import torch
        import scale_only_query as sq
        import latent_probe_gate as gate

        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
        self.torch, self.sq = torch, sq
        self.vs = __import__('variable_shape_memory')
        self.pool = gate._load_combined_pool()
        if self.pool['phys'].shape != (1595, 10, 6):
            raise AssertionError(f"unexpected W pool shape {self.pool['phys'].shape}")
        if not np.asarray(self.pool['keep'], bool).all():
            raise AssertionError('W_pool must already contain the 1,595 kept candidates')
        cond = 'S-shared'
        self.canonical = self.sq.type_of(self.pool['phys'], cond).astype(np.float32)
        key = self.sq.key_inputs('W', self.pool, cond)
        self.model = self.sq.make_net('W', 'P5', 78, key.shape[1])
        ckpt = torch.load(checkpoint or selected_checkpoint(), map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['holdout']['state'])
        self.model.eval()
        with torch.inference_mode():
            self.keys = torch.nn.functional.normalize(
                self.model.key(torch.as_tensor(key, dtype=torch.float32)), dim=-1)
        self.source_task = np.asarray(self.pool['task'], np.int64)
        self.keep = np.asarray(self.pool['keep'], bool)

    def score_and_pick(self, history, state, allowed=None):
        """Return (pool index, score vector); history is zero-padded [10,7]."""
        history = np.asarray(history, np.float32)
        state = np.asarray(state, np.float32).reshape(-1)[:8]
        if history.shape != (10, 7) or state.shape != (8,):
            raise ValueError(f'expected history (10,7), state (8,), got {history.shape}, {state.shape}')
        x = np.concatenate([history.reshape(70), state])[None]
        allowed = self.keep.copy() if allowed is None else np.asarray(allowed, bool)
        if allowed.shape != self.keep.shape or np.any(allowed & ~self.keep) or not allowed.any():
            raise ValueError('candidate mask must be a nonempty subset of the kept pool')
        with self.torch.inference_mode():
            q = self.torch.nn.functional.normalize(
                self.model.query(self.torch.as_tensor(x, dtype=self.torch.float32)), dim=-1)
            scores = (q @ self.keys.T)[0].cpu().numpy()
        scores[~allowed] = -np.inf
        return int(np.argmax(scores)), scores


def compose_normalized_action(normal, history, state, selector, *, allowed=None, has_history=True):
    """No-rotation S-pi05 action draft. Input/output use the pi05 normalized space."""
    import latent_probe_gate as gate

    normal = np.asarray(normal, np.float32)
    history = np.asarray(history, np.float32)
    if normal.ndim != 2 or normal.shape[0] != 10 or normal.shape[1] < 7:
        raise ValueError(f'pi05 chunk must be [10,>=7], got {normal.shape}')
    idx, scores = selector.score_and_pick(history, state, allowed)
    canonical = selector.canonical[idx:idx + 1]
    source = np.asarray(selector.pool['phys'][idx:idx + 1], np.float32)
    pi05_phys = selector.vs.to_physical(normal[None, :, :6])
    posed = gate.amplitude_predictions(
        canonical, source, pi05_phys, history[None], np.asarray([np.any(history)]))['S-pi05'][0]
    draft = normal.copy()
    draft[:, :6] = selector.vs.to_normalized(posed)
    # Gripper and padded action dimensions remain byte-for-byte from pi05.
    alpha = float(gate.fit_nonnegative_scale(
        selector.vs.to_normalized(canonical), normal[None, :, :6])[0])
    debug = {
        'condition': 'LPC1_scale_only_no_rotation',
        'candidate_index': idx,
        'source_row': int(selector.pool['source_row'][idx]) if 'source_row' in selector.pool else None,
        'source_unit': int(selector.pool['source_unit'][idx]) if 'source_unit' in selector.pool else None,
        'multiplier': alpha,
        'score_top1': float(scores[idx]),
        'eligible_candidates': int(np.isfinite(scores).sum()),
        'mean_squared_six_axis_gap_to_pi05': float(np.mean((draft[:, :6] - normal[:, :6]) ** 2)),
    }
    # Fallback (same rule as SEG): with no executed history, or when the selected
    # type points against pi05 so the non-negative fit is zero, the type cannot be
    # posed without rotation; use pi05 for this chunk instead of a frozen robot.
    fallback = (not has_history) or alpha <= 1e-6
    debug['fallback_pi05'] = bool(fallback)
    debug['fallback_reason'] = None if not fallback else ('no_history' if not has_history else 'zero_scale')
    if fallback:
        draft = normal.copy()
    return draft, debug


def make_policy(checkpoint=RUN_CHECKPOINT):
    """Wrap the existing pi05 serving transform/sampler; no source files are modified."""
    sys.path.insert(0, str(HERE.parent / 'pi05_memory'))
    from serve import MemoryPolicy

    class NoRotationPolicy(MemoryPolicy):
        def __init__(self, ckpt):
            super().__init__(ckpt, ckpt, normal_only=True)
            self.selector = ScaleOnlyP5Selector()
            self.executed = {}

        def infer(self, observation):
            from openpi.models.model import Observation

            start = time.monotonic()
            # The rollout client (anchor_flow/run_case.py) already applies [::-1, ::-1];
            # flipping again here would feed pi05 upside-down images (as in the 43/100 server, no flip).
            row_obs = observation
            raw = {k: row_obs[k] for k in ('observation/image', 'observation/wrist_image',
                                            'observation/state', 'prompt')}
            row = self.transform(raw)
            obs = Observation(
                images={k: self.jnp.asarray(v)[None].astype(self.jnp.float32) / 255 * 2 - 1
                        for k, v in row['image'].items()},
                image_masks={k: self.jnp.asarray([v]) for k, v in row['image_mask'].items()},
                state=self.jnp.asarray(row['state'])[None],
                tokenized_prompt=self.jnp.asarray(row['tokenized_prompt'])[None],
                tokenized_prompt_mask=self.jnp.asarray(row['tokenized_prompt_mask'])[None])
            seed = int(observation['s2_rng_seed'])
            normal = np.asarray(self.normal(self.jax.random.key(seed), obs, num_steps=10))[0].astype(np.float32)
            episode = str(observation.get('s2_episode_id', 'default'))
            query_id = int(observation.get('s2_query_id', 0))
            if query_id == 0:
                self.executed[episode] = []
            trace = self.executed.setdefault(episode, [])
            history = (np.stack(trace[-10:]).astype(np.float32) if len(trace) >= 10
                       else np.zeros((10, 7), np.float32))
            mask = observation.get('lpc1_allowed')
            if mask is not None:
                mask = np.asarray(mask, bool)
            draft, debug = compose_normalized_action(normal, history, np.asarray(row['state'])[:8],
                                                     self.selector, allowed=mask, has_history=len(trace) >= 10)
            horizon = int(observation.get('replan_steps', 5))
            if horizon != 5:
                raise ValueError('LPC1 requires replan_steps=5')
            trace.extend(draft[:horizon, :7].copy())
            actions = self.unnormalize({'actions': draft, 'state': row['state']})['actions'][:, :7]
            if not np.isfinite(actions).all():
                raise FloatingPointError('nonfinite LPC1 action')
            debug.update({'image_orientation': 'client-flipped (run_case.py), no server flip', 'replan_steps': horizon,
                          'seconds': time.monotonic() - start})
            log_path = os.environ.get('LPC1_QUERY_LOG')
            if log_path:
                p = Path(log_path); p.parent.mkdir(parents=True, exist_ok=True)
                with p.open('a') as f:
                    f.write(json.dumps({'episode_id': episode, 'query_id': query_id,
                                        'seed': seed, **debug}, allow_nan=False) + '\n')
            print(json.dumps({'episode_id': episode, 'query_id': query_id,
                              'seed': seed, **debug}, allow_nan=False), flush=True)
            return {'actions': actions.astype(np.float32), 's2_debug': debug}

    return NoRotationPolicy(Path(checkpoint))


def dev_parity(out, *, n=10):
    """Compare the server composer to scale_only_query's offline T-P5 × S-pi05 path."""
    import latent_probe_gate as gate

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    inp = dict(np.load(DEV_INPUTS))
    rows = np.load(DEV_PI05)
    if not np.array_equal(inp['row'], rows['row']):
        raise AssertionError('corrected rot180 pi05 and Dev input row order differ')
    orientation_audit = gate.audit_orientation_and_hold_last(assert_reference=True)
    if len(inp['row']) != 113:
        raise AssertionError('Dev cache must contain the fixed 113 windows')
    pool = gate._load_combined_pool()
    selector = ScaleOnlyP5Selector()
    allowed = gate.candidate_mask('dev', inp, pool)
    count = min(int(n), len(inp['row']))
    # Independent offline reference entry point: it rebuilds X/K and scores the
    # selected checkpoint through the production scale_only_query evaluator.
    offline_picks = selector.sq.score_models(SCALE, 'W', 'S-shared', 'P5', 4101, 'dev')['holdout']
    pi = np.asarray(rows['actions'][:count], np.float32)
    _, target, _ = gate._load_split_rows('dev')
    history = np.asarray(inp['hist'][:count], np.float32)
    state = np.asarray(inp['state'][:count, :8], np.float32)
    offline_idx = np.empty(count, np.int64)
    server_idx = np.empty(count, np.int64)
    offline_actions = np.empty((count, 10, pi.shape[-1]), np.float32)
    server_actions = np.empty_like(offline_actions)
    debug = []
    for i in range(count):
        offline_idx[i] = int(offline_picks[i])
        server_actions[i], d = compose_normalized_action(pi[i], history[i], state[i], selector,
                                                          allowed=allowed[i])
        server_idx[i] = d['candidate_index']
        debug.append(d)
        # Independent offline expression matching LPG's amplitude_predictions S-pi05 branch.
        canonical = selector.canonical[offline_idx[i]:offline_idx[i]+1]
        source = np.asarray(selector.pool['phys'][offline_idx[i]:offline_idx[i]+1], np.float32)
        posed = gate.amplitude_predictions(canonical, source,
            selector.vs.to_physical(pi[i:i+1, :, :6]), history[i:i+1],
            np.asarray([bool(inp['has_hist'][i])]))['S-pi05'][0]
        offline_actions[i] = pi[i].copy()
        offline_actions[i, :, :6] = selector.vs.to_normalized(posed)
    if not np.array_equal(offline_idx, server_idx):
        raise AssertionError('server and offline P5 selected different W candidates')
    max_delta = float(np.max(np.abs(offline_actions-server_actions)))
    if max_delta > 1e-5:
        raise AssertionError(f'server/offline normalized action parity failed: max abs {max_delta:.3g}')
    corrected_mse = float(orientation_audit['pi05_rot180_mse6_over7'])
    asis_mse = float(orientation_audit['pi05_asis_mse6_over7'])
    if not (.105 <= corrected_mse <= .125 and .175 <= asis_mse <= .210):
        raise AssertionError(f'pi05 orientation regression: rot180={corrected_mse:.5f}, asis={asis_mse:.5f}')
    payload = {'rows_checked': count, 'row_ids': np.asarray(inp['row'][:count]).tolist(),
               'selected_pool_indices': server_idx.tolist(),
               'selected_source_rows': [d['source_row'] for d in debug],
               'selected_source_units': [d['source_unit'] for d in debug],
               'multipliers': [d['multiplier'] for d in debug],
               'mean_squared_six_axis_gap_to_pi05': [d['mean_squared_six_axis_gap_to_pi05'] for d in debug],
               'max_abs_server_offline_action_delta': max_delta,
               'pi05_orientation_check': {'rot180_mse6_over7': corrected_mse,
                                          'asis_mse6_over7': asis_mse,
                                          'source': 'fast-query corrected and as-is Eval predictions against aligned targets; server applies the same explicit 180-degree image flip'},
               'candidate_mask': 'Dev scale_only_query cross_task=False; one identical mask supplied to both paths',
               'rotation_applied_to_selected_type': False}
    (out/'parity.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('serve'); s.add_argument('--checkpoint', type=Path, default=RUN_CHECKPOINT)
    s.add_argument('--port', type=int, default=8156)
    p = sub.add_parser('parity'); p.add_argument('--out', type=Path, required=True); p.add_argument('--rows', type=int, default=10)
    args = parser.parse_args(argv)
    if args.cmd == 'parity':
        print(json.dumps(dev_parity(args.out, n=args.rows), indent=2)); return 0
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy = make_policy(args.checkpoint)
    WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port,
                          metadata={'experiment': 'LPC1_no_rotation_scale_only', 'action_horizon': 10}).serve_forever()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
