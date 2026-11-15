"""Single-trial EX1 integration smoke: frozen KEY2-F P5 retrieval + Expert bridge."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory')]


class EX1SmokePolicy:
    def __init__(self, pi05_checkpoint: Path, p5_checkpoint: Path, expert_checkpoint: Path,
                 t0: float = .3):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        import orbax.checkpoint as ocp
        import torch
        from openpi.shared.nnx_utils import PathRegex, module_jit
        import expert_ex1 as ex1
        import key_shared_latent as key2
        import skill_expert_refine as exp3
        from openpi import transforms
        from openpi.models.model import ModelType
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.policies.libero_policy import LiberoInputs
        from openpi.shared.normalize import load

        torch.set_num_threads(int(__import__('os').environ.get('OMP_NUM_THREADS', '2')))
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(.30)
        self.torch, self.jax, self.jnp = torch, jax, jnp
        self.key2 = key2
        base = exp3._module(exp3.PI05 / 'run.py', 'ex1_smoke_expert_policy')
        self.model, _ = base.load_policy(pi05_checkpoint)
        # Keep a genuinely frozen π0.5 action head for the normal reference
        # and gripper channel; the Expert-restored model is used only by bridge.
        ref_base = exp3._module(exp3.PI05 / 'run.py', 'ex1_smoke_reference_policy')
        self.reference_model, _ = ref_base.load_policy(pi05_checkpoint)
        self.normal = module_jit(self.reference_model.sample_actions, static_argnames=('num_steps',))
        filt = nnx.All(nnx.Param, PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
        with ocp.PyTreeCheckpointer() as loader:
            partial = loader.restore(Path(expert_checkpoint))
        state = nnx.state(self.model, filt)
        state.replace_by_pure_dict(jax.tree.map(jnp.asarray, partial))
        nnx.update(self.model, state)
        self.bridge = module_jit(self.model.sample_bridge, static_argnames=('num_steps',))
        self.prefix = module_jit(self.model.prefix_features)
        normfiles = list((pi05_checkpoint / 'assets').rglob('norm_stats.json'))
        if len(normfiles) != 1:
            raise ValueError(f'ambiguous normalization files: {normfiles}')
        stats = load(normfiles[0].parent)
        self.transform = transforms.compose([
            LiberoInputs(ModelType.PI05), transforms.Normalize(stats, use_quantiles=True),
            transforms.ResizeImages(224, 224), transforms.TokenizePrompt(PaligemmaTokenizer(200), discrete_state_input=False),
            transforms.PadStatesAndActions(32)])
        self.unnormalize = transforms.Unnormalize(stats, use_quantiles=True)
        self.t0 = float(t0)

        self.bank, self.bank_input, _, _, _, _, _, _ = key2._build_key2_queries(
            exp3.RUN, ex1.KEY2_MISSING, ex1.KEY2_SEEN, ex1.KEY2_UNSEEN, ex1.KEY2_VJEPA_EVAL)
        self.p5 = key2._new_model('p5').cuda().eval()
        self.p5.load_state_dict(torch.load(p5_checkpoint, map_location='cuda', weights_only=True))
        self.candidate_reps, _, _ = key2._encode_candidates(
            self.p5, 'p5', self.bank_input, self.bank, torch.device('cuda'), batch=16)
        self.candidate_reps = torch.as_tensor(self.candidate_reps, device='cuda', dtype=torch.float32)
        self.logit_scale = key2._logit_scale_parameter(self.p5).detach().exp().clamp(max=key2.LOGIT_SCALE_MAX)
        self.units = exp3.unit_window(self.bank['natural_actions'])
        self.histories: dict[str, list[np.ndarray]] = {}

    def infer(self, observation):
        from openpi.models.model import Observation
        started = time.monotonic()
        raw = {k: observation[k] for k in ('observation/image', 'observation/wrist_image',
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
        key = self.jax.random.key(seed)
        normal = np.asarray(self.normal(key, obs, num_steps=10), np.float32)

        ep = str(observation['s2_episode_id'])
        qid = int(observation['s2_query_id']) % 1000
        if qid == 0:
            self.histories[ep] = []
        executed = self.histories.setdefault(ep, [])
        history = np.zeros((10, 7), np.float32)
        history_mask = np.zeros(10, bool)
        prev = executed[-10:]
        if prev:
            history[-len(prev):] = np.asarray(prev, np.float32)
            history_mask[-len(prev):] = True

        hidden, groups = self.prefix(obs)
        hidden, groups = np.asarray(hidden)[0], np.asarray(groups)[0]
        valid = groups >= 0
        pooled = hidden[valid].mean(0, dtype=np.float32)[None]
        state8 = np.asarray(obs.state, np.float32)[0, :8][None]
        part = {'pooled': pooled, 'history': history[None], 'history_mask': history_mask[None], 'state': state8}
        with self.torch.no_grad():
            qrep = self.key2._model_batch_encode(self.p5, 'p5', part, self.torch.device('cuda'))
            scores = self.torch.einsum('qkd,nkd->qn', qrep, self.candidate_reps)
            scores = scores / qrep.shape[1] * self.logit_scale
            unit_id = int(scores[0].argmax().item())

        draft = np.zeros_like(normal)
        draft[0, :, :6] = self.units[unit_id, :, :6]
        draft[0, :, 6] = normal[0, :, 6]
        eps_key = self.jax.random.fold_in(key, 7)
        eps = self.jax.random.normal(eps_key, (1, 10, 32))
        result = self.bridge(self.jax.random.key(0), obs, self.jnp.asarray(draft),
                             self.jnp.asarray([self.t0]), num_steps=10, noise=eps)
        normalized = np.asarray(result, np.float32)
        executed.extend(normalized[0, :5, :7].copy())
        physical = self.unnormalize({'actions': normalized[0], 'state': row['state']})['actions'][:, :7]
        if not np.isfinite(physical).all():
            raise FloatingPointError('nonfinite EX1 action')
        source = self.bank['source'][unit_id]
        task, episode, start, end = map(int, source)
        return {'actions': physical.astype(np.float32), 's2_debug': {
            'unit_key': [task, episode, start, end], 'p5_score': float(scores[0, unit_id].item()),
            't0': self.t0, 'history_valid_steps': int(history_mask.sum()),
            'seconds': time.monotonic() - started}}


def main():
    import os
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    p = argparse.ArgumentParser()
    p.add_argument('--pi05-checkpoint', type=Path, required=True)
    p.add_argument('--p5-checkpoint', type=Path, required=True)
    p.add_argument('--expert-checkpoint', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--t0', type=float, default=.3)
    a = p.parse_args()
    policy = EX1SmokePolicy(a.pi05_checkpoint, a.p5_checkpoint, a.expert_checkpoint, a.t0)
    WebsocketPolicyServer(policy, host='127.0.0.1', port=a.port,
                          metadata={'experiment': 'EX1_actual_P5_smoke', 'action_horizon': 10}).serve_forever()


if __name__ == '__main__':
    main()
