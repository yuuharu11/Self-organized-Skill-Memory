#!/usr/bin/env python3
"""Policy server for confidence-adaptive skill-correction Expert."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
PI05=HERE.parent
sys.path.insert(0,str(PI05))
from bprior.serve import BPriorPolicy, retrieve_b  # noqa: E402
_spec=importlib.util.spec_from_file_location('pi05_skill_expert_run_module',HERE/'run.py')
_run=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_run)
load_expert,seed_values,confidence_mode=_run.load_expert,_run.seed_values,_run.confidence_mode


class SkillExpertPolicy:
    def __init__(self, run:Path, output:Path, checkpoint:Path, condition:str):
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
        os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','0.20')
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        from openpi.shared.nnx_utils import module_jit,PathRegex
        self.jax,self.jnp=jax,jnp
        self.run,self.output,self.condition=run,output,condition
        self.base=BPriorPolicy(run,checkpoint,run=run,verify=True)
        self.model=self.base.model
        self.normal=self.base.base_policy.normal
        filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
        self.filter=filt
        self.official=nnx.state(self.model,filt).to_pure_dict()
        # Validate the official fallback on a fixed cached row before overlay.
        from run import obs_batch,load_rows
        rows=load_rows(run);self.rows=rows
        probe=obs_batch(rows,np.asarray([0],np.int64))
        probe_noise=jnp.zeros((1,10,32),jnp.float32)
        self.official_probe=np.asarray(self.normal(jax.random.key(19),probe,num_steps=10,noise=probe_noise))
        load_expert(self.model,Path(json.loads((output/'best_bridge.json').read_text())['path']))
        self.trained=nnx.state(self.model,filt).to_pure_dict()
        # A weight swap back to the frozen action expert must reproduce the
        # official path exactly, including the same explicit noise tensor.
        state=nnx.state(self.model,filt);state.replace_by_pure_dict(self.official);nnx.update(self.model,state)
        fallback_probe=np.asarray(self.normal(jax.random.key(19),probe,num_steps=10,noise=probe_noise))
        np.testing.assert_array_equal(self.official_probe,fallback_probe)
        state=nnx.state(self.model,filt);state.replace_by_pure_dict(self.trained);nnx.update(self.model,state)
        self.bridge=module_jit(self.model.sample_bridge,static_argnames=('num_steps',))
        edges_path=output/'confidence_edges.json'
        self.edges=json.loads(edges_path.read_text())
        if condition not in {'C','A','B','D'}: raise ValueError(condition)
        self._checks={'normal_fallback_exact':True,
            'official_probe_sha256':hashlib.sha256(self.official_probe.tobytes()).hexdigest(),
            'bprior_check':self.base.cache_check,
            'seed_rule':'env=7+init_state_id; query=(7+init_state_id)*100000+query_number',
            'confidence_edges':self.edges,'condition':condition}

    def _with_official(self, fn):
        import flax.nnx as nnx
        state=nnx.state(self.model,self.filter);state.replace_by_pure_dict(self.official);nnx.update(self.model,state)
        try: return fn()
        finally:
            state=nnx.state(self.model,self.filter);state.replace_by_pure_dict(self.trained);nnx.update(self.model,state)

    def infer(self,observation):
        from openpi.models.model import Observation
        started=time.monotonic()
        raw={k:observation[k] for k in ['observation/image','observation/wrist_image','observation/state','prompt']}
        row=self.base.transform(raw)
        obs=Observation(images={k:self.jnp.asarray(v)[None].astype(self.jnp.float32)/255*2-1 for k,v in row['image'].items()},
            image_masks={k:self.jnp.asarray([v]) for k,v in row['image_mask'].items()},
            state=self.jnp.asarray(row['state'])[None],tokenized_prompt=self.jnp.asarray(row['tokenized_prompt'])[None],
            tokenized_prompt_mask=self.jnp.asarray(row['tokenized_prompt_mask'])[None])
        # run_case's wire format carries the init index in the episode key and
        # uses the query number as the low five decimal digits of rng_seed.
        episode_parts=str(observation.get('s2_episode_id','')).split(':')
        init_id=int(episode_parts[-1]) if len(episode_parts)>=3 else 0
        seed=int(observation['s2_rng_seed'])
        query_no=int(seed % 100000)
        expected_env,expected_query=seed_values(init_id,query_no)
        if seed != expected_query: raise AssertionError(f'query seed mismatch: got {seed} expected {expected_query}')
        key=self.jax.random.key(seed)
        hidden,groups=self.base.features(obs);hidden,groups=np.asarray(hidden)[0],np.asarray(groups)[0]
        feature=hidden[groups>=0].mean(0)[None].astype(np.float32)
        state=np.asarray(obs.state,np.float32)
        q_b,q_r=self.base.codec.prior(feature,state)
        memory_id,score,local=retrieve_b(q_b,self.base.memory_b,self.base.memory_ids,
            mean=self.base.b_mean,std=self.base.b_std,unit_bank=self.base.memory_b_unit)
        mode=confidence_mode(float(score),self.edges,self.condition)
        if mode['fallback']:
            result=self._with_official(lambda:self.normal(key,obs,num_steps=10))
            draft=None
        else:
            if mode['draft']=='zero': draft=np.zeros((1,10,32),np.float32)
            else: draft=self.base.codec.decode(self.base.memory_b[local:local+1],np.asarray(q_r),feature,state)
            result=self.bridge(key,obs,self.jnp.asarray(draft),self.jnp.asarray([mode['t0']]),num_steps=10)
        normalized=np.asarray(result)[0]
        actions=self.base.unnormalize({'actions':normalized,'state':row['state']})['actions'][:,:7].astype(np.float32)
        if not np.isfinite(actions).all(): raise FloatingPointError('nonfinite policy output')
        debug={'condition':self.condition,'score':float(score),'memory_row':memory_id,'fallback':mode['fallback'],
            't0':mode['t0'],'draft':mode['draft'],'num_steps':10,'env_seed_expected':expected_env,
            'query_seed':seed,'query_number':query_no,'normal_fallback_exact':mode['fallback'],
            'seconds':time.monotonic()-started}
        return {'actions':actions,'s2_debug':debug}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--condition',choices=['C','A','B','D'],required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--status-dir',type=Path,required=True)
    a=p.parse_args();a.status_dir.mkdir(parents=True,exist_ok=True)
    policy=SkillExpertPolicy(a.run,a.output,a.checkpoint,a.condition)
    (a.status_dir/'server_checks.json').write_text(json.dumps(policy._checks,indent=2)+'\n')
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,
       metadata={'experiment':'pi05_memory40_skill_expert','condition':a.condition,'action_horizon':10,'num_steps':10}).serve_forever()

if __name__=='__main__': main()
