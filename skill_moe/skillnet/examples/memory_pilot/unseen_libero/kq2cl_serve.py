#!/usr/bin/env python3
"""KQ2-CL runtime server.

mode memory (A / B / A-const):
  1. history tokens for tau = t-9..t: [a_{tau-1}, s_tau, a-valid] from the states and actions the
     client actually executed (sent by kq2cl_closed.py; q01/q99 normalization as kq2_data.qnorm);
  2. (iii) only: 34 pi0.5 VLM tokens of the current observation (vq_sources._vq_features on the frozen
     pi0.5: 4x4-pooled normalized image outputs of base+wrist, prompt mean, prompt last; fp16 as KQ2);
  3. KQ2 query -> logits over the fixed K-x1 keys of the 4,516 LIBERO-40 units -> top-1 unit;
  4. heads (kq2cl_heads bundle): ê (confidence head) and the 10-step gripper decoder;
  5. ê > b -> pi0.5 chunk (fallback);  else draft = top-1 stored arm window + hold-last gripper,
     t0 = clip(c*sqrt(ê), lo, hi) (or a constant), KQ2-CL Expert bridge; arm from the Expert,
     gripper from the decoder (switch once at the first step whose opposite probability > th).
mode pi05: plain pi0.5 sample_actions (same rng key rule as EKQ).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory')]
import kq2cl_common as C  # noqa: E402


class KQ2CLPolicy:
    def __init__(self, *, mode: str, query: str | None, expert_checkpoint: Path | None, bundle: Path | None,
                 tpol: dict | None):
        import torch, jax, jax.numpy as jnp, flax.nnx as nnx, orbax.checkpoint as ocp
        from openpi.shared.nnx_utils import PathRegex, module_jit
        from openpi import transforms
        from openpi.models.model import ModelType, Observation
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.policies.libero_policy import LiberoInputs
        from openpi.shared.normalize import load
        import skill_expert_refine as exp3
        import expert_ex3 as ex3
        import vq_sources as vs
        import kq2_data as kd
        self.torch, self.jax, self.jnp, self.kd = torch, jax, jnp, kd
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(os.environ.get('TORCH_MEM_FRACTION', '.1')))
        self.device = torch.device('cuda')
        self.mode, self.query = mode, query
        run = exp3._module(HERE.parent / 'pi05_memory/run.py', 'kq2cl_serve_ref')
        self.ref, _ = run.load_policy(C.PI05)
        self.normal = module_jit(self.ref.sample_actions, static_argnames=('num_steps',))
        normfiles = list(C.PI05.joinpath('assets').rglob('norm_stats.json'))
        stats = load(normfiles[0].parent)
        self.transform = transforms.compose([LiberoInputs(ModelType.PI05), transforms.Normalize(stats, use_quantiles=True),
                                             transforms.ResizeImages(224, 224), transforms.TokenizePrompt(PaligemmaTokenizer(200), discrete_state_input=False),
                                             transforms.PadStatesAndActions(32)])
        self.unnormalize = transforms.Unnormalize(stats, use_quantiles=True); self.Observation = Observation
        self.dump = os.environ.get('KQ2CL_DUMP')
        if mode == 'pi05':
            return
        self.vlm = module_jit(types.MethodType(vs._vq_features, self.ref)) if query == 'iii' else None
        self.model, _ = exp3._module(HERE.parent / 'pi05_memory/run.py', 'kq2cl_serve_expert').load_policy(C.PI05)
        filt = nnx.All(nnx.Param, PathRegex(ex3.PARAM_REGEX))
        with ocp.PyTreeCheckpointer() as c:
            partial = c.restore(Path(expert_checkpoint))
        st = nnx.state(self.model, filt); st.replace_by_pure_dict(jax.tree.map(jnp.asarray, partial)); nnx.update(self.model, st)
        self.bridge = module_jit(self.model.sample_bridge, static_argnames=('num_steps',))
        self.qnet, self.qcfg = C.load_query(query, self.device)
        if bool(self.qcfg['xattn']) != (query == 'iii'):
            raise AssertionError('query / VLM mismatch')
        self.keys_np = np.load(C.KEYS).astype(np.float32)
        self.keys = torch.as_tensor(self.keys_np, device=self.device)
        self.units = np.load(C.ROOT / 'key-motion-01/final/data/bank_actions.npy').astype(np.float32)
        self.unit_task = np.load(C.ROOT / 'key-motion-01/final/data/bank_task.npy')
        if len(self.units) != 4516 or self.keys_np.shape != (4516, 256):
            raise AssertionError('bank / keys')
        b = torch.load(bundle, map_location='cpu', weights_only=False)
        self.ckind, self.gkind = b['confidence']['kind'], b['gripper']['kind']
        self.conf = C.make_head(self.ckind, 1); self.conf.load_state_dict(b['confidence']['state_dict']); self.conf.eval()
        self.grip = C.make_head(self.gkind, 10, extra=4); self.grip.load_state_dict(b['gripper']['state_dict']); self.grip.eval()
        self.th = float(b['gripper_th'])
        self.tpol = tpol
        self.scale = float(self.qnet.logit_scale.exp().clamp(max=100.))

    # -- helpers -----------------------------------------------------------
    def _obs(self, observation):
        jnp = self.jnp
        row = self.transform({k: observation[k] for k in ('observation/image', 'observation/wrist_image', 'observation/state', 'prompt')})
        obs = self.Observation(images={k: jnp.asarray(v)[None].astype(jnp.float32) / 255 * 2 - 1 for k, v in row['image'].items()},
                               image_masks={k: jnp.asarray([v]) for k, v in row['image_mask'].items()}, state=jnp.asarray(row['state'])[None],
                               tokenized_prompt=jnp.asarray(row['tokenized_prompt'])[None], tokenized_prompt_mask=jnp.asarray(row['tokenized_prompt_mask'])[None])
        return row, obs

    def tokens(self, observation):
        """History window from the executed states/actions sent by the client."""
        sh = np.asarray(observation['kq2/state_hist'], np.float32).reshape(-1, 8)      # s_{t-n+1..t}
        ah = np.asarray(observation['kq2/action_hist'], np.float32).reshape(-1, 7)     # a_{t-m..t-1}
        t = int(observation['kq2/t'])
        sn = self.kd.qnorm(sh, 'state'); an = self.kd.qnorm(ah, 'actions') if len(ah) else np.zeros((0, 7), np.float32)
        seq_s = np.zeros((t + 1, 8), np.float32); seq_a = np.zeros((max(t, 1), 7), np.float32)
        seq_s[t + 1 - len(sn):] = sn
        if len(an):
            seq_a[t - len(an):t] = an
        return C.window_tokens(seq_a, seq_s, t), sn, an, t

    def infer(self, observation):
        started = time.monotonic(); torch = self.torch; jax = self.jax
        row, obs = self._obs(observation)
        key = jax.random.key(int(observation['s2_rng_seed']))
        if self.mode == 'pi05':
            out = np.asarray(self.normal(key, obs, num_steps=10), np.float32)
            physical = self.unnormalize({'actions': out[0], 'state': row['state']})['actions'][:, :7]
            return {'actions': physical.astype(np.float32), 's2_debug': {'runtime': 'kq2cl_pi05', 'fallback_pi05': True,
                                                                          'gripper_chunk': [float(x) for x in out[0, :, 6]],
                                                                          'seconds': time.monotonic() - started}}
        (x, valid), sn, an, t = self.tokens(observation)
        vlm = None
        if self.vlm is not None:
            f = {k: np.asarray(v) for k, v in self.vlm(obs).items()}
            vlm = np.zeros((1, 34, 2048), np.float16)
            vlm[0, :32] = f['vlm_img'][0]; vlm[0, 32] = f['vlm_txt_mean'][0].astype(np.float16); vlm[0, 33] = f['vlm_txt_last'][0].astype(np.float16)
        T = lambda a, dt=torch.float32: torch.as_tensor(a, device=self.device, dtype=dt)
        split = C.InlineSplit(T(x[None]), T(valid[None], torch.bool), None if vlm is None else T(vlm, torch.float16))
        with torch.no_grad():
            q = self.qnet(split, T([0], torch.long))
            logits = self.qnet.score(q, self.keys).double().cpu().numpy()
        qn = q.float().cpu().numpy()
        summ = C.summarize_scores(logits, np.ones_like(logits, bool))
        feats = C.head_features(qn, summ['ids'], summ['logits'], summ['entropy'], summ['top1_prob'], summ['top5_mass'],
                                self.keys_np, self.scale)
        last_valid = len(an) > 0
        last_cmd = float(an[-1, 6]) if last_valid else C.DEFAULT_GRIP
        last_sign = 1. if (last_valid and last_cmd > 0) else -1.
        gx = C.grip_extra(np.asarray([last_cmd]), np.asarray([last_valid]), sn[-1:, 6:8])
        with torch.no_grad():
            ehat = float(np.exp(self.conf(torch.as_tensor(C.head_input(feats, self.ckind))).numpy()[0, 0]))
            pclose = 1 / (1 + np.exp(-self.grip(torch.as_tensor(C.head_input(feats, self.gkind, gx))).numpy()))
        grip = C.decode_gripper(pclose, np.asarray([last_sign]), self.th)[0]
        unit = int(summ['ids'][0, 0])
        tp = self.tpol
        fallback = tp.get('b') is not None and ehat > float(tp['b'])
        if 'const' in tp:
            t0 = float(tp['const'])
        else:
            t0 = float(np.clip(tp['c'] * np.sqrt(ehat), tp['lo'], tp['hi']))
        if fallback:
            out = np.asarray(self.normal(key, obs, num_steps=10), np.float32).copy()
        else:
            draft = np.zeros((1, 10, 32), np.float32); draft[0, :, :6] = self.units[unit, :, :6]; draft[0, :, 6] = last_sign
            noise = jax.random.normal(jax.random.fold_in(key, 7), (1, 10, 32))
            out = np.asarray(self.bridge(jax.random.key(0), obs, self.jnp.asarray(draft), self.jnp.asarray([t0]), num_steps=10,
                                         noise=noise), np.float32).copy()
            out[0, :, 6] = grip
        physical = self.unnormalize({'actions': out[0], 'state': row['state']})['actions'][:, :7]
        if not np.isfinite(physical).all():
            raise FloatingPointError('nonfinite KQ2-CL action output')
        dbg = {'runtime': 'kq2cl', 'query': self.query, 'unit_id': unit, 'unit_task': int(self.unit_task[unit]),
               'confidence_error': ehat, 't0': t0, 'fallback_pi05': bool(fallback), 'step_t': t,
               'history_valid_steps': int(valid.sum()), 'draft_gripper': last_sign, 'gripper_chunk': [float(v) for v in out[0, :, 6]],
               'decoder_pclose': [float(v) for v in pclose[0]], 'top1_logit': float(summ['logits'][0, 0]),
               'gap2': float(summ['logits'][0, 0] - summ['logits'][0, 1]), 'entropy': float(summ['entropy'][0]),
               'seconds': time.monotonic() - started}
        if self.dump:
            d = Path(self.dump); d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{str(observation['s2_episode_id']).replace(':', '_')}_q{int(observation['s2_query_id']) % 1000:03d}.npz",
                                state_hist=np.asarray(observation['kq2/state_hist']), action_hist=np.asarray(observation['kq2/action_hist']),
                                t=t, x=x, valid=valid, vlm=vlm if vlm is not None else np.zeros(0), q=qn, top_ids=summ['ids'],
                                top_logits=summ['logits'], ehat=ehat, pclose=pclose, image=np.asarray(observation['observation/image']),
                                wrist=np.asarray(observation['observation/wrist_image']), prompt=str(observation['prompt']),
                                raw_state=np.asarray(observation['observation/state']))
        return {'actions': physical.astype(np.float32), 's2_debug': dbg}


def main():
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=('memory', 'pi05'))
    p.add_argument('--query', choices=('iii', 'i')); p.add_argument('--expert-checkpoint', type=Path)
    p.add_argument('--bundle', type=Path); p.add_argument('--tpol', default='{}', help='JSON {c,lo,hi,b} or {const,b}')
    p.add_argument('--port', type=int, required=True)
    a = p.parse_args()
    policy = KQ2CLPolicy(mode=a.mode, query=a.query, expert_checkpoint=a.expert_checkpoint, bundle=a.bundle, tpol=json.loads(a.tpol))
    WebsocketPolicyServer(policy, host='127.0.0.1', port=a.port,
                          metadata={'experiment': 'KQ2-CL', 'mode': a.mode, 'query': a.query, 'horizon': 10}).serve_forever()


if __name__ == '__main__':
    main()
