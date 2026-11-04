"""Policy server for saved-state counterfactual branches (smoke/full harness)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent
sys.path.insert(0, str(PI05))
from fast_search import FastTokenIndex


class BranchPolicy:
    def __init__(self, run: Path, checkpoint: Path):
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        import torch
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get('BRANCH_TORCH_MEM_FRACTION', '.20')), 0)
        import jax
        import jax.numpy as jnp
        self.jax, self.jnp = jax, jnp
        spec = importlib.util.spec_from_file_location('pi05_memory_branch_base', PI05/'serve.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.TokenIndex = FastTokenIndex
        self.base = module.MemoryPolicy(run, checkpoint, normal_only=False)
        self.calls = 0

    def infer(self, request):
        from openpi.models.model import Observation
        import time
        started = time.monotonic()
        raw = {k: request[k] for k in ['observation/image','observation/wrist_image',
                                       'observation/state','prompt']}
        row = self.base.transform(raw)
        obs = Observation(
            images={k:self.jnp.asarray(v)[None].astype(self.jnp.float32)/255*2-1
                    for k,v in row['image'].items()},
            image_masks={k:self.jnp.asarray([v]) for k,v in row['image_mask'].items()},
            state=self.jnp.asarray(row['state'])[None],
            tokenized_prompt=self.jnp.asarray(row['tokenized_prompt'])[None],
            tokenized_prompt_mask=self.jnp.asarray(row['tokenized_prompt_mask'])[None])
        seed = int(request['s2_rng_seed'])
        key = self.jax.random.key(seed)
        condition = request.get('branch_condition', 'normal_rank')
        debug = {'condition':condition,'rng_seed':seed}
        if condition in ('normal_rank','current'):
            if condition == 'normal_rank':
                # Keep the normal path byte-identical to ordinary pi05_memory
                # inference; the diagnostic retrieval is computed afterwards.
                result = self.base.normal(key, obs, num_steps=10)
            hidden, groups = self.base.features(obs)
            hidden, groups = np.asarray(hidden)[0], np.asarray(groups)[0]
            top = self.base.index.search_topk(hidden, groups, 8)
            debug['top8'] = [{'memory_row':r,'score':s} for r,s in top]
            feature = hidden[groups >= 0].mean(0)[None]
            state = np.asarray(obs.state)
            if condition == 'normal_rank':
                debug['top1_memory_row'] = int(top[0][0])
            else:
                memory_row, score = top[0]
                from calibrate import apply_schedule
                mode = int(apply_schedule(np.asarray([score]), self.base.schedule)[0])
                if mode == 0:
                    result = self.base.normal(key, obs, num_steps=10)
                    debug.update(memory_row=int(memory_row),score=float(score),mode=mode,
                                 fallback=True,t0=None)
                else:
                    t0 = (1.0,0.6,0.4,0.2)[mode]
                    result = self._bridge(key, obs, int(memory_row), feature, state, t0)
                    debug.update(memory_row=int(memory_row),score=float(score),mode=mode,
                                 fallback=False,t0=t0)
        elif condition in ('normal','normal_alt'):
            result = self.base.normal(key, obs, num_steps=10)
        elif condition == 'zero':
            draft = np.zeros((1,10,32), np.float32)
            result = self.base.bridge(key, obs, self.jnp.asarray(draft),
                                      self.jnp.asarray([0.6]), num_steps=10)
            debug.update(t0=0.6,draft='zero')
        elif condition == 'memory_row':
            memory_row = int(request['memory_row'])
            t0 = float(request['branch_t0'])
            hidden, groups = self.base.features(obs)
            hidden, groups = np.asarray(hidden)[0], np.asarray(groups)[0]
            feature = hidden[groups >= 0].mean(0)[None]
            state = np.asarray(obs.state)
            result = self._bridge(key, obs, memory_row, feature, state, t0)
            debug.update(memory_row=memory_row,t0=t0)
        else:
            raise ValueError(f'unknown branch_condition {condition!r}')
        actions = np.asarray(result)[0]
        actions = self.base.unnormalize({'actions':actions,'state':row['state']})['actions'][:,:7]
        actions = actions.astype(np.float32)
        debug['seconds'] = time.monotonic()-started
        return {'actions':actions,'s2_debug':debug}

    def _bridge(self,key,obs,memory_row,feature,state,t0):
        _, residual = self.base.codec.prior(feature,state)
        draft = self.base.codec.decode(np.asarray(self.base.bases[memory_row])[None],
                                       residual,feature,state)
        return self.base.bridge(key,obs,self.jnp.asarray(draft),
                                self.jnp.asarray([t0]),num_steps=10)


def main():
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--port',type=int,required=True)
    args=parser.parse_args()
    policy=BranchPolicy(args.run,args.checkpoint)
    WebsocketPolicyServer(policy,host='127.0.0.1',port=args.port,
        metadata={'experiment':'pi05_memory40_rescue_branching','action_horizon':10,'num_steps':10}).serve_forever()


if __name__=='__main__':
    main()
