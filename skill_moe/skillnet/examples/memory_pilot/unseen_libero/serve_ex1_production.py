#!/usr/bin/env python3
"""EX1 production websocket policy for actual KEY2-F retrieved units."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
PI05=HERE.parent/'pi05_memory'
import sys
sys.path[:0]=[str(HERE),str(PI05)]


class EX1Policy:
    def __init__(self, mode, *, candidate_cache, p5_checkpoint, pi05_checkpoint,
                 expert_checkpoint=None, t0=0.2):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        import orbax.checkpoint as ocp
        import torch
        from openpi.shared.nnx_utils import PathRegex,module_jit
        import key_shared_latent as key2
        import skill_expert_refine as exp3
        import expert_ex1 as ex1
        from openpi import transforms
        from openpi.models.model import ModelType,Observation
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.policies.libero_policy import LiberoInputs
        from openpi.shared.normalize import load

        mode_to_ranker={'D0_P0_top1':'p0','D1_P5_top1':'p5','D3_E_VLM_H_top1':'e_vlm_h'}
        if mode not in mode_to_ranker:raise ValueError(mode)
        ranker_mode=mode_to_ranker[mode]
        self.torch,self.jax,self.jnp,self.key2,self.ex1=torch,jax,jnp,key2,ex1
        self.mode=mode;self.k=1
        torch.set_num_threads(int(__import__('os').environ.get('OMP_NUM_THREADS','2')))
        if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(.18)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type!='cuda':raise RuntimeError('EX1 production serving requires CUDA')
        self.reference_model,_=exp3._module(PI05/'run.py','ex1_prod_reference').load_policy(pi05_checkpoint)
        self.normal=module_jit(self.reference_model.sample_actions,static_argnames=('num_steps',))
        self.prefix=module_jit(self.reference_model.prefix_features)
        self.model=None;self.bridge=None;self.t0=float(t0)
        if expert_checkpoint:
            self.model,_=exp3._module(PI05/'run.py','ex1_prod_expert').load_policy(pi05_checkpoint)
            filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
            with ocp.PyTreeCheckpointer() as loader:partial=loader.restore(Path(expert_checkpoint))
            state=nnx.state(self.model,filt);state.replace_by_pure_dict(jax.tree.map(jnp.asarray,partial));nnx.update(self.model,state)
            self.bridge=module_jit(self.model.sample_bridge,static_argnames=('num_steps',))
        key2_config=ex1.load_locked_key2_checkpoints()
        self.ranker_mode=ranker_mode
        ranker=key2._new_model(ranker_mode).to(self.device)
        ranker.load_state_dict(torch.load(key2_config[ranker_mode]['checkpoint'],map_location=self.device,weights_only=True))
        ranker.eval();self.ranker=ranker
        with np.load(candidate_cache,allow_pickle=False) as z:
            reps_key={'p0':'candidate_reps_p0','p5':'candidate_reps','e_vlm_h':'candidate_reps_e_vlm_h'}[ranker_mode]
            self.candidate_reps=torch.as_tensor(z[reps_key].copy(),device=self.device,dtype=torch.float32)
            self.units=np.asarray(z['unit_windows'],np.float32)
            self.unit_task=np.asarray(z['task'],np.int32)
            self.unit_source=np.asarray(z['source'],np.int32)
            self.logit_scale=float(z[{'p0':'p0_logit_scale','p5':'logit_scale','e_vlm_h':'e_vlm_h_logit_scale'}[ranker_mode]])
        if self.candidate_reps.shape[0]!=len(self.units) or self.units.shape[1:]!=(10,7):
            raise ValueError('EX1 closed-loop candidate cache shape mismatch')
        normfiles=list((Path(pi05_checkpoint)/'assets').rglob('norm_stats.json'))
        if len(normfiles)!=1:raise ValueError(f'ambiguous checkpoint norm stats: {normfiles}')
        stats=load(normfiles[0].parent)
        self.transform=transforms.compose([LiberoInputs(ModelType.PI05),transforms.Normalize(stats,use_quantiles=True),
            transforms.ResizeImages(224,224),transforms.TokenizePrompt(PaligemmaTokenizer(200),discrete_state_input=False),
            transforms.PadStatesAndActions(32)])
        self.unnormalize=transforms.Unnormalize(stats,use_quantiles=True)
        self.Observation=Observation;self.histories={}

    def infer(self,observation):
        start=time.monotonic()
        raw={k:observation[k] for k in ('observation/image','observation/wrist_image','observation/state','prompt')}
        row=self.transform(raw)
        obs=self.Observation(images={k:self.jnp.asarray(v)[None].astype(self.jnp.float32)/255*2-1 for k,v in row['image'].items()},
            image_masks={k:self.jnp.asarray([v]) for k,v in row['image_mask'].items()},
            state=self.jnp.asarray(row['state'])[None],tokenized_prompt=self.jnp.asarray(row['tokenized_prompt'])[None],
            tokenized_prompt_mask=self.jnp.asarray(row['tokenized_prompt_mask'])[None])
        seed=int(observation['s2_rng_seed']);key=self.jax.random.key(seed)
        normal=np.asarray(self.normal(key,obs,num_steps=10),np.float32)
        episode=str(observation['s2_episode_id']);qid=int(observation['s2_query_id'])%1000
        if qid==0:self.histories[episode]=[]
        executed=self.histories.setdefault(episode,[])
        hist=np.zeros((10,7),np.float32);mask=np.zeros(10,bool);prev=executed[-10:]
        if prev:hist[-len(prev):]=np.asarray(prev,np.float32);mask[-len(prev):]=True
        hidden,groups=self.prefix(obs);hidden=np.asarray(hidden)[0];groups=np.asarray(groups)[0]
        valid=groups>=0;pooled=hidden[valid].mean(0,dtype=np.float32)[None]
        part={'pooled':pooled,'tokens':hidden[None],'groups':groups[None],
              'history':hist[None],'history_mask':mask[None],
              'state':np.asarray(obs.state,np.float32)[0,:8][None]}
        with self.torch.no_grad():
            q=self.key2._model_batch_encode(self.ranker,self.ranker_mode,part,self.device)
            if q.ndim==3:
                scores=(self.torch.einsum('qkd,nkd->qn',q,self.candidate_reps)/q.shape[1])*self.logit_scale
            else:scores=(q@self.candidate_reps.T)*self.logit_scale
            ids=self.torch.argsort(scores[0],descending=True,stable=True)[:1].cpu().numpy()
            arm=self.units[int(ids[0]),:,:6].copy()
        draft=np.zeros_like(normal);draft[0,:,:6]=arm;draft[0,:,6]=normal[0,:,6]
        use_expert=self.model is not None and observation.get('s2_condition')!='EX1_D2_draft_only'
        if use_expert:
            eps=self.jax.random.normal(self.jax.random.fold_in(key,7),(1,10,32))
            normalized=np.asarray(self.bridge(self.jax.random.key(0),obs,self.jnp.asarray(draft),
                self.jnp.asarray([self.t0]),num_steps=10,noise=eps),np.float32)
        else:normalized=draft
        # EX1 trains only the arm-six velocity loss. Keep the gripper entirely
        # from the frozen π0.5 reference, including the executed history.
        normalized=self.ex1.restore_pi05_gripper(normalized,normal)
        executed.extend(normalized[0,:5,:7].copy())
        physical=self.unnormalize({'actions':normalized[0],'state':row['state']})['actions'][:,:7]
        if not np.isfinite(physical).all():raise FloatingPointError('EX1 returned nonfinite action')
        i=int(ids[0])
        return {'actions':physical.astype(np.float32),'s2_debug':{'mode':self.mode,'draft_only':not use_expert,
            'unit_id':i,'unit_task':int(self.unit_task[i]),'unit_source':self.unit_source[i].astype(int).tolist(),
            'top_unit_ids':ids.astype(int).tolist(),'top_scores':scores[0,ids].detach().cpu().tolist(),
            't0':self.t0 if use_expert else 0.,'history_valid':int(mask.sum()),
            'draft_final_arm_mse':float(np.mean((normalized[0,:,:6]-draft[0,:,:6])**2)),
            'seconds':time.monotonic()-start}}


def main():
    import argparse
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('D0_P0_top1','D1_P5_top1','D3_E_VLM_H_top1'),required=True)
    p.add_argument('--candidate-cache',type=Path,required=True);p.add_argument('--p5-checkpoint',type=Path,required=True)
    p.add_argument('--pi05-checkpoint',type=Path,required=True);p.add_argument('--expert-checkpoint',type=Path)
    p.add_argument('--t0',type=float,default=.2);p.add_argument('--port',type=int,required=True)
    a=p.parse_args();policy=EX1Policy(a.mode,candidate_cache=a.candidate_cache,p5_checkpoint=a.p5_checkpoint,
        pi05_checkpoint=a.pi05_checkpoint,expert_checkpoint=a.expert_checkpoint,t0=a.t0)
    WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,
        metadata={'experiment':'EX1 actual locked P5; DCT3 raw action unit','mode':a.mode,'horizon':10}).serve_forever()

if __name__=='__main__':main()
