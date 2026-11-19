#!/usr/bin/env python3
"""EX3 runtime server: KQ-final top-1 draft -> EX3 Expert (arm + gripper). No π0.5 anywhere.

* retrieval: KQ-final query (C1a attention / C1b history+state MLP) over fixed K-x1 keys,
  4,516 LIBERO-40 units (same scoring path as serve_expert_ex2.py KQ_* modes).
* draft: stored arm window of the top-1 unit + hold-last gripper (sign of the last executed
  gripper, open (-1) at the episode start).
* t0: clip(c*sqrt(ê), lo, hi), ê from the EKQ π0.5-free confidence head (frozen copy in
  expert-ex3-01/ekq-frozen), (c, lo, hi) chosen on Dev for EX3.
* output: the EX3 Expert's 10-step chunk; gripper executed as the sign of the Expert channel.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; PI05_DIR = HERE.parent / 'pi05_memory'
sys.path[:0] = [str(HERE), str(PI05_DIR)]


class EX3Policy:
    def __init__(self, *, query: str, expert_checkpoint: Path, tmap: dict):
        import torch, jax, jax.numpy as jnp, flax.nnx as nnx, orbax.checkpoint as ocp
        from openpi.shared.nnx_utils import PathRegex, module_jit
        from openpi import transforms
        from openpi.models.model import ModelType, Observation
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.policies.libero_policy import LiberoInputs
        from openpi.shared.normalize import load
        import key_final as kf
        import expert_kq_runtime as ekq
        import skill_expert_refine as exp3
        import expert_ex3 as ex3
        self.torch, self.jax, self.jnp, self.kf, self.ekq, self.ex3 = torch, jax, jnp, kf, ekq, ex3
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(os.environ.get('TORCH_MEM_FRACTION', '.18')))
        self.device = torch.device('cuda')
        self.query = query; self.map = {k: float(tmap[k]) for k in ('c', 'lo', 'hi')}
        # Expert (π0.5 architecture, EX3 partial weights). module_jit is built *after* loading.
        self.model, _ = exp3._module(PI05_DIR / 'run.py', 'ex3_serve').load_policy(ex3.PI05)
        filt = nnx.All(nnx.Param, PathRegex(ex3.PARAM_REGEX))
        with ocp.PyTreeCheckpointer() as c:
            partial = c.restore(Path(expert_checkpoint))
        st = nnx.state(self.model, filt); st.replace_by_pure_dict(jax.tree.map(jnp.asarray, partial)); nnx.update(self.model, st)
        self.bridge = module_jit(self.model.sample_bridge, static_argnames=('num_steps',))
        # Retrieval
        blob = torch.load(ex3.QUERIES[query], map_location='cpu', weights_only=False); self.cfg = blob['cfg']
        self.ranker = kf.make_query(self.cfg).to(self.device); self.ranker.load_state_dict(blob['query']); self.ranker.eval()
        self.units = np.load(ex3.KQF / 'data/bank_actions.npy').astype(np.float32)
        self.unit_task = np.load(ex3.KQF / 'data/bank_task.npy'); self.unit_source = np.load(ex3.KQF / 'data/bank__source.npy')
        self.keys = torch.as_tensor(np.load(ex3.KQF / 'data/x1_keys.npy'), device=self.device, dtype=torch.float32)
        if tuple(self.keys.shape) != (len(self.units), 256) or len(self.units) != 4516:
            raise AssertionError('unexpected K-x1 key / unit bank shape')
        c = torch.load(ex3.FROZEN / f'{query}_confidence.pt', map_location='cpu', weights_only=False)
        self.conf = ekq.new_conf_head(c['kind']); self.conf.load_state_dict(c['state_dict']); self.conf.eval()
        normfiles = list(ex3.PI05.joinpath('assets').rglob('norm_stats.json'))
        stats = load(normfiles[0].parent)
        self.transform = transforms.compose([LiberoInputs(ModelType.PI05), transforms.Normalize(stats, use_quantiles=True),
                                             transforms.ResizeImages(224, 224), transforms.TokenizePrompt(PaligemmaTokenizer(200), discrete_state_input=False),
                                             transforms.PadStatesAndActions(32)])
        self.unnormalize = transforms.Unnormalize(stats, use_quantiles=True); self.Observation = Observation
        self.histories: dict[str, list] = {}

    def infer(self, observation):
        started = time.monotonic(); jnp = self.jnp; torch = self.torch
        row = self.transform({k: observation[k] for k in ('observation/image', 'observation/wrist_image', 'observation/state', 'prompt')})
        obs = self.Observation(images={k: jnp.asarray(v)[None].astype(jnp.float32) / 255 * 2 - 1 for k, v in row['image'].items()},
                               image_masks={k: jnp.asarray([v]) for k, v in row['image_mask'].items()}, state=jnp.asarray(row['state'])[None],
                               tokenized_prompt=jnp.asarray(row['tokenized_prompt'])[None], tokenized_prompt_mask=jnp.asarray(row['tokenized_prompt_mask'])[None])
        key = self.jax.random.key(int(observation['s2_rng_seed']))
        episode = str(observation['s2_episode_id']); qid = int(observation['s2_query_id']) % 1000
        if qid == 0:
            self.histories[episode] = []
        executed = self.histories.setdefault(episode, [])
        hist = np.zeros((10, 7), np.float32); mask = np.zeros(10, bool); prev = executed[-10:]
        if prev:
            hist[-len(prev):] = np.asarray(prev, np.float32); mask[-len(prev):] = True
        st = np.asarray(obs.state, np.float32)[:, :8]
        feature = np.concatenate([hist.reshape(1, -1), mask.astype(np.float32).reshape(1, -1), st], axis=1)
        T = lambda a, dt=torch.float32: torch.as_tensor(a, device=self.device, dtype=dt)
        with torch.no_grad():
            q = self.ranker(history=T(hist[None]), history_mask=T(mask[None], torch.bool), state=T(st), hs=T(feature))
            scores = self.kf.score(q, self.keys, 'dot', self.ranker.logit_scale)
        score_np = scores.double().cpu().numpy(); qnorm = q.float().norm(dim=-1).cpu().numpy()
        summ = self.ekq.score_summary(score_np); unit = int(summ['ids'][0, 0])
        feats = self.ekq.confidence_features(summ['ids'], summ['vals'], summ['entropy'], summ['top1_prob'], self.units,
                                             hist[None], mask[None], st, qnorm)
        errorhat = float(self.ekq.predict_conf(self.conf, feats)[0])
        t0 = float(self.ex3.select_t0(errorhat, self.map['c'], self.map['lo'], self.map['hi']))
        grip0 = float(self.ex3.hold_last_gripper(hist[None, 9, 6], mask[None].any(1))[0, 0])
        draft = np.zeros((1, 10, 32), np.float32); draft[0, :, :6] = self.units[unit, :, :6]; draft[0, :, 6] = grip0
        noise = self.jax.random.normal(self.jax.random.fold_in(key, 7), (1, 10, 32))
        out = np.asarray(self.bridge(self.jax.random.key(0), obs, jnp.asarray(draft), jnp.asarray([t0]), num_steps=10, noise=noise), np.float32).copy()
        raw_grip = out[0, :, 6].copy()
        out[0, :, 6] = np.where(raw_grip >= 0, 1., -1.)
        executed.extend(out[0, :5, :7].copy())
        physical = self.unnormalize({'actions': out[0], 'state': row['state']})['actions'][:, :7]
        if not np.isfinite(physical).all():
            raise FloatingPointError('nonfinite EX3 action output')
        return {'actions': physical.astype(np.float32), 's2_debug': {
            'runtime': 'ex3', 'query': self.query, 'unit_id': unit, 'unit_task': int(self.unit_task[unit]),
            'unit_source': np.asarray(self.unit_source[unit]).astype(int).tolist(), 'confidence_error': errorhat, 't0': t0,
            'fallback_pi05': False, 'pi05_sampled': False, 'history_valid_steps': int(mask.sum()), 'draft_gripper': grip0,
            'gripper_chunk': [float(x) for x in out[0, :, 6]], 'gripper_raw': [float(x) for x in raw_grip],
            'top1_score': float(score_np[0, unit]), 'score_margin12': float(summ['vals'][0, 0] - summ['vals'][0, 1]),
            'seconds': time.monotonic() - started}}


def main():
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--query', required=True, choices=('C1a', 'C1b')); p.add_argument('--expert-checkpoint', type=Path, required=True)
    p.add_argument('--map', required=True, help='JSON {"c":..,"lo":..,"hi":..} chosen on Dev'); p.add_argument('--port', type=int, required=True)
    a = p.parse_args()
    policy = EX3Policy(query=a.query, expert_checkpoint=a.expert_checkpoint, tmap=json.loads(a.map))
    WebsocketPolicyServer(policy, host='127.0.0.1', port=a.port,
                          metadata={'experiment': 'EX3 retrained Expert (arm+gripper), no π0.5', 'query': a.query, 'horizon': 10}).serve_forever()


if __name__ == '__main__':
    main()
