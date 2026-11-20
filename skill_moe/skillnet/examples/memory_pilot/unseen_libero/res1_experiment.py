"""RES1 K1 residual Expert hooks, resumable context cache, and smoke runner.

This module does not start production training while KQ2-CL owns the GPU.  It
provides the exact bridge objective and an action-expert hidden-state hook so
the reviewer can inspect the integration before the gated run.
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
sys.path.insert(0, str(HERE))
import kq2cl_common as C
from res1_core import draw_rank_samples

OUT = C.ROOT / 'res1-01'
KQ_DONE = C.OUT / 'DONE'


def attach_delta_attention(model, *, dim: int = 256, rng_seed: int = 4101):
    """Attach a zero-gated one-head NNX cross-attention module to a K1 model.

    The method is called after the pretrained K1 policy has been restored.
    NNX child modules are attached under ``res1_delta`` and can be included in
    the action-expert trainable filter using ``.*res1_delta.*``.
    """
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx

    class DeltaXAttn(nnx.Module):
        def __init__(self):
            rngs = nnx.Rngs(rng_seed)
            self.q = nnx.Linear(1024, dim, rngs=rngs)
            self.k = nnx.Linear(dim, dim, rngs=rngs)
            self.v = nnx.Linear(dim, dim, rngs=rngs)
            self.o = nnx.Linear(dim, 1024, rngs=rngs)
            self.state = nnx.Linear(8, dim, rngs=rngs)
            self.gate = nnx.Param(jnp.zeros((), jnp.float32))

        def __call__(self, expert_tokens, saved, current, valid):
            # saved/current: {h:[B,1,256], state:[B,1,8], vlm?:[B,34,256]}
            s = [saved['h'][:, None, :], self.state(saved['state'])[:, None, :]]
            c = [current['h'][:, None, :], self.state(current['state'])[:, None, :]]
            masks = [jnp.ones((len(expert_tokens), 2), jnp.bool_)]
            if 'vlm' in saved or 'vlm' in current:
                if 'vlm' not in saved or 'vlm' not in current:
                    raise ValueError('saved/current VLM context mismatch')
                s.append(saved['vlm']); c.append(current['vlm'])
                vm = valid if valid is not None else jnp.ones(saved['vlm'].shape[:2], jnp.bool_)
                masks.append(vm)
            s, c = jnp.concatenate(s, axis=1), jnp.concatenate(c, axis=1)
            mem = jnp.concatenate((s, c, c - s), axis=1)
            ok = jnp.concatenate(masks, axis=1)
            ok = jnp.concatenate((ok, ok, ok), axis=1)
            q, k, v = self.q(expert_tokens), self.k(mem), self.v(mem)
            logits = jnp.einsum('btd,bnd->btn', q, k) / jnp.sqrt(float(dim))
            logits = jnp.where(ok[:, None], logits, -1e9)
            weights = jax.nn.softmax(logits, axis=-1)
            cross = self.o(jnp.einsum('btn,bnd->btd', weights, v))
            return expert_tokens + jnp.tanh(self.gate.value) * cross

    model.res1_delta = DeltaXAttn()
    return model


def res1_trainable_filter():
    """KQ2-CL K1 action-expert leaves plus the new RES1-X layer."""
    import flax.nnx as nnx
    from openpi.shared.nnx_utils import PathRegex
    return nnx.All(nnx.Param, PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj|res1_delta).*'))


def suffix_hidden_and_velocity(model, observation, full_action, time, *, prefix_context=None):
    """K1 velocity with exact suffix hidden tokens exposed before action_out_proj.

    This is a local copy of MemoryPi05.velocity's forward path.  It leaves the
    base method untouched and provides the insertion point for RES1-X.
    """
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask

    obs = model_api.preprocess_observation(None, observation, train=False)
    if prefix_context is None:
        prefix, mask, ar = model.embed_prefix(obs)
        _, cache = model.PaliGemma.llm([prefix, None], mask=make_attn_mask(mask, ar),
                                       positions=jnp.cumsum(mask, axis=1) - 1)
    else:
        mask, cache = prefix_context
    suffix, smask, sar, cond = model.embed_suffix(obs, full_action, time)
    attn = jnp.concatenate([jnp.broadcast_to(mask[:, None, :],
                            (len(full_action), suffix.shape[1], mask.shape[1])),
                            make_attn_mask(smask, sar)], axis=-1)
    pos = jnp.sum(mask, axis=1)[:, None] + jnp.cumsum(smask, axis=1) - 1
    (_, hidden), _ = model.PaliGemma.llm([None, suffix], mask=attn, positions=pos,
                                         kv_cache=cache, adarms_cond=[None, cond])
    hidden = hidden[:, -model.action_horizon:]
    return hidden, model.action_out_proj(hidden)


def res1_velocity(model, observation, draft, residual_state, progress,
                  *, saved_context=None, current_context=None, context_valid=None,
                  prefix_context=None, use_delta=False):
    """Compute residual velocity from K1 action expert, with optional RES1-X.

    ``draft`` is [B,10,32], ``residual_state`` [B,10,6]. Other draft channels
    (including gripper) are passed through unchanged. Expert Q are the actual
    suffix hidden states from K1, not a proxy made from actions.
    """
    import jax.numpy as jnp
    full = jnp.asarray(draft)
    full = full.at[..., :6].set(full[..., :6] + residual_state)
    if use_delta:
        if not hasattr(model, 'res1_delta'):
            raise ValueError('RES1-X module has not been attached')
        if saved_context is None or current_context is None:
            raise ValueError('RES1-X requires saved and current context')
        hidden, _ = suffix_hidden_and_velocity(model, observation, full, progress,
                                               prefix_context=prefix_context)
        corrected = model.res1_delta(hidden, saved_context, current_context, context_valid)
        return model.action_out_proj(corrected)[..., :6]
    # K1 velocity has been initialized from the pretrained K1 expert and is
    # fine-tuned against the residual flow target.
    return model.velocity(observation, full, progress, prefix_context=prefix_context)[..., :6]


def jax_flow_batch(model, observation, actions, draft, noise, progress, *,
                   saved_context=None, current_context=None, context_valid=None,
                   prefix_context=None, use_delta=False):
    """JAX differentiable arm-only residual flow matching objective."""
    import jax.numpy as jnp
    target = jnp.asarray(actions)[..., :6] - jnp.asarray(draft)[..., :6]
    noise = jnp.asarray(noise)[..., :6]
    u = jnp.asarray(progress).reshape(-1, 1, 1)
    x = (1. - u) * target + u * noise
    v_target = noise - target
    pred = res1_velocity(model, observation, draft, x, progress,
                         saved_context=saved_context, current_context=current_context,
                         context_valid=context_valid, prefix_context=prefix_context,
                         use_delta=use_delta)
    return jnp.mean(jnp.square(pred - v_target))


def jax_regression_batch(model, observation, actions, draft, *, saved_context=None,
                         current_context=None, context_valid=None, prefix_context=None,
                         use_delta=False):
    """Direct residual regression comparison using the same K1 action expert."""
    import jax.numpy as jnp
    target = jnp.asarray(actions)[..., :6] - jnp.asarray(draft)[..., :6]
    zeros = jnp.zeros_like(target)
    progress = jnp.zeros((len(target),), jnp.float32)
    pred = res1_velocity(model, observation, draft, zeros, progress,
                         saved_context=saved_context, current_context=current_context,
                         context_valid=context_valid, prefix_context=prefix_context,
                         use_delta=use_delta)
    return jnp.mean(jnp.square(pred - target))


def make_nnx_train_step(*, use_delta: bool, objective: str = 'flow'):
    """Build the optimizer-independent JIT step for K1/RES1/RES1-X production.

    The caller owns the AdamW state/checkpoint loop and can use the same
    checkpoint cadence/optimizer as KQ2-CL.  ``batch`` dictionary entries are
    NNX/JAX arrays. Loss is only on the six arm channels.
    """
    if objective not in ('flow', 'regression'):
        raise ValueError('objective must be flow or regression')
    import flax.nnx as nnx
    train_filter = res1_trainable_filter()

    @nnx.jit
    def step(model, observation, actions, draft, noise, progress,
             saved_context, current_context, context_valid):
        if objective == 'flow':
            def loss_fn(m):
                return jax_flow_batch(m, observation, actions, draft, noise, progress,
                                      saved_context=saved_context, current_context=current_context,
                                      context_valid=context_valid, use_delta=use_delta)
        else:
            def loss_fn(m):
                return jax_regression_batch(m, observation, actions, draft,
                                            saved_context=saved_context, current_context=current_context,
                                            context_valid=context_valid, use_delta=use_delta)
        return nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, train_filter))(model)
    return step


def integrate_k1_residual(velocity_fn, observation, draft, noise, *, steps=10,
                          saved_context=None, current_context=None, context_valid=None,
                          prefix_context=None):
    """JAX deterministic integration from white noise residual to correction."""
    import jax
    import jax.numpy as jnp
    if steps <= 0:
        raise ValueError('steps must be positive')
    r = jnp.asarray(noise)[..., :6]
    def body(i, value):
        t = jnp.full((len(value),), 1. - i / steps, jnp.float32)
        v = velocity_fn(observation, draft, value, t, saved_context=saved_context,
                        current_context=current_context, context_valid=context_valid,
                        prefix_context=prefix_context)
        return value - v / steps
    return jnp.asarray(draft)[..., :6] + jax.lax.fori_loop(0, steps, body, r)


def check_kq2_gate() -> None:
    if not KQ_DONE.exists():
        raise RuntimeError(f'KQ2-CL has not written completion marker: {KQ_DONE}; production GPU run is gated')


def smoke(steps: int = 32, seed: int = 4101) -> dict:
    """CPU smoke: residual FM + regression + zero-gate delta attention updates."""
    import torch
    from torch import nn
    from res1_core import DeltaContextCrossAttention, residual_flow_loss, residual_regression_loss
    torch.manual_seed(seed); torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 8)))
    device = torch.device('cpu')
    class Toy(nn.Module):
        def __init__(self, delta=False):
            super().__init__(); self.inp = nn.Linear(13, 64); self.out = nn.Linear(64, 6)
            self.delta = DeltaContextCrossAttention(64, 4) if delta else None
        def forward(self, obs, point, u, saved, current):
            tok = torch.cat((point, obs[:, None, :].expand(-1, point.shape[1], -1)), -1)
            h = torch.tanh(self.inp(tok))
            if self.delta is not None:
                h = self.delta(h, saved, current)
            return self.out(h)
    b,t=4,10
    obs=torch.randn(b,7); draft=torch.randn(b,t,32); actions=draft.clone(); actions[...,:6]+=0.2*torch.randn(b,t,6)
    saved=torch.randn(b,3,64); current=saved+0.1*torch.randn_like(saved); noise=torch.randn(b,t,6)
    models={k:Toy(k=='xattn').to(device) for k in ('flow','xattn','reg')}
    opts={k:torch.optim.Adam(m.parameters(),lr=3e-3) for k,m in models.items()}
    losses={k:[] for k in models}
    for i in range(steps):
        u=torch.rand(b)
        for name,m in models.items():
            if name=='reg':
                pred=m(obs,torch.zeros_like(noise),torch.zeros(b),None,None)
                loss=residual_regression_loss(pred,actions,draft)
            else:
                def vf(_obs, point, prog, sc, cc, model=m): return model(_obs,point,prog,sc,cc)
                sc,cc=(saved,current) if name=='xattn' else (None,None)
                loss=residual_flow_loss(vf,obs,actions,draft,noise,u,saved_context=sc,current_context=cc)
            opts[name].zero_grad();loss.backward();opts[name].step();losses[name].append(float(loss.detach()))
    ratio=float(np.linalg.norm((actions-draft)[...,:6].numpy())/max(np.linalg.norm(draft[...,:6].numpy()),1e-8))
    report={'status':'smoke_complete','steps':steps,'seed':seed,'device':'cpu',
            'initial_loss':{k:v[0] for k,v in losses.items()},'final_loss':{k:v[-1] for k,v in losses.items()},
            'residual_to_draft_global_norm_ratio':ratio,
            'delta_gate':float(torch.tanh(models['xattn'].delta.gate).detach()),
            'k1_suffix_hidden_insertion':'implemented in suffix_hidden_and_velocity; full checkpoint smoke gated by GPU owner'}
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--smoke-steps',type=int,default=32)
    ap.add_argument('--extract-context',choices=('i','iii'));ap.add_argument('--output',type=Path,default=OUT)
    ap.add_argument('--production',action='store_true');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.production: check_kq2_gate()
    if a.extract_context:
        from res1_context import extract_bank_contexts
        p=extract_bank_contexts(a.extract_context,a.output/'context')
        print(json.dumps({'status':'context_complete','path':str(p),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}))
    if a.smoke:
        r=smoke(a.smoke_steps);(a.output/'smoke.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(r))
    if a.production:
        raise RuntimeError('KQ2-CL completion marker exists; invoke reviewed train/eval stage after KQ2 GPU lease releases')


if __name__=='__main__':main()

