#!/usr/bin/env python3
"""Runtime server for EX2 motion-only Expert with confidence-conditioned t0."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;PI05=HERE.parent/'pi05_memory'
sys.path[:0]=[str(HERE),str(PI05)]


class EX2Policy:
    def __init__(self, *, mode, candidate_cache, retrieval_cache, schedule, confidence,
                 pi05_checkpoint, expert_checkpoint, p5_checkpoint):
        import torch,jax,jax.numpy as jnp,flax.nnx as nnx,orbax.checkpoint as ocp
        from openpi.shared.nnx_utils import PathRegex,module_jit
        from openpi import transforms
        from openpi.models.model import ModelType,Observation
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.policies.libero_policy import LiberoInputs
        from openpi.shared.normalize import load
        import key_shared_latent as key2
        import key_grid
        import expert_ex1 as ex1
        import skill_expert_refine as exp3
        import expert_ex2 as ex2
        if mode not in ('D0_P0_top1','D1_P5_top1','D3_E_VLM_H_top1','D5_HS_top1'):raise ValueError(mode)
        self.mode=mode;self.torch=torch;self.jax=jax;self.jnp=jnp;self.key2=key2;self.key_grid=key_grid;self.ex1=ex1;self.ex2=ex2
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','1')))
        if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(float(os.environ.get('TORCH_MEM_FRACTION','.18')))
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type!='cuda':raise RuntimeError('EX2 production server requires CUDA')
        schedule=json.loads(Path(schedule).read_text());self.schedule=schedule['schedule']
        self.conf=torch.load(confidence,map_location='cpu',weights_only=True)
        self.reg=ex2._new_confidence_model();self.reg.load_state_dict(self.conf['state_dict']);self.reg.eval()
        self.reference_model,_=exp3._module(PI05/'run.py','ex2_prod_reference').load_policy(pi05_checkpoint)
        self.normal=module_jit(self.reference_model.sample_actions,static_argnames=('num_steps',))
        self.prefix=module_jit(self.reference_model.prefix_features)
        self.model,_=exp3._module(PI05/'run.py','ex2_prod_expert').load_policy(pi05_checkpoint)
        filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
        with ocp.PyTreeCheckpointer() as loader:partial=loader.restore(Path(expert_checkpoint))
        state=nnx.state(self.model,filt);state.replace_by_pure_dict(jax.tree.map(jnp.asarray,partial));nnx.update(self.model,state)
        self.bridge=module_jit(self.model.sample_bridge,static_argnames=('num_steps',))
        keyconfig=ex1.load_locked_key2_checkpoints();self.ranker_mode={'D0_P0_top1':'p0','D1_P5_top1':'p5','D3_E_VLM_H_top1':'e_vlm_h'}.get(mode)
        with np.load(candidate_cache,allow_pickle=False) as z:
            self.units=np.asarray(z['unit_windows'],np.float32);self.unit_task=np.asarray(z['task'],np.int32);self.unit_source=np.asarray(z['source'],np.int32)
            if mode=='D5_HS_top1':
                d5=json.loads((Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-grid-01/production-04/results.json')).read_text())['new_models']['history_state']['variants']['two_tower_mlp']
                self.ranker=key_grid.PairMLPRanker(88,shared=False).to(self.device)
                self.ranker.load_state_dict(torch.load(d5['checkpoint'],map_location=self.device,weights_only=True));self.ranker.eval()
                self.candidate_vec=torch.as_tensor(np.load(Path(retrieval_cache)/'candidate_history_state.npy',mmap_mode='r').copy(),device=self.device,dtype=torch.float32)
                with torch.no_grad():self.candidate_keys=self.ranker.key(self.candidate_vec)
                self.logit_scale=float(self.ranker.logit_scale.detach().exp().clamp(max=key2.LOGIT_SCALE_MAX).cpu())
            else:
                repkey={'p0':'candidate_reps_p0','p5':'candidate_reps','e_vlm_h':'candidate_reps_e_vlm_h'}[self.ranker_mode]
                self.candidate_reps=torch.as_tensor(z[repkey].copy(),device=self.device,dtype=torch.float32)
                self.ranker=key2._new_model(self.ranker_mode).to(self.device)
                self.ranker.load_state_dict(torch.load(keyconfig[self.ranker_mode]['checkpoint'],map_location=self.device,weights_only=True));self.ranker.eval()
                self.logit_scale=float(z[{'p0':'p0_logit_scale','p5':'logit_scale','e_vlm_h':'e_vlm_h_logit_scale'}[self.ranker_mode]])
        if len(self.units)!=4516:raise AssertionError(f'EX2 candidate bank changed: {self.units.shape}')
        normfiles=list(Path(pi05_checkpoint).joinpath('assets').rglob('norm_stats.json'))
        if len(normfiles)!=1:raise ValueError(f'expected one norm_stats.json, got {normfiles}')
        stats=load(normfiles[0].parent)
        self.transform=transforms.compose([LiberoInputs(ModelType.PI05),transforms.Normalize(stats,use_quantiles=True),transforms.ResizeImages(224,224),
            transforms.TokenizePrompt(PaligemmaTokenizer(200),discrete_state_input=False),transforms.PadStatesAndActions(32)])
        self.unnormalize=transforms.Unnormalize(stats,use_quantiles=True);self.Observation=Observation
        self.histories={};self.draft_scale=float(self.schedule['scale_c']);self.threshold=self.schedule['fallback_error_threshold']
        self.t0_low,self.t0_high=map(float,self.schedule['clip'])

    def infer(self,observation):
        started=time.monotonic();raw={k:observation[k] for k in ('observation/image','observation/wrist_image','observation/state','prompt')}
        row=self.transform(raw)
        obs=self.Observation(images={k:self.jnp.asarray(v)[None].astype(self.jnp.float32)/255*2-1 for k,v in row['image'].items()},
            image_masks={k:self.jnp.asarray([v]) for k,v in row['image_mask'].items()},state=self.jnp.asarray(row['state'])[None],
            tokenized_prompt=self.jnp.asarray(row['tokenized_prompt'])[None],tokenized_prompt_mask=self.jnp.asarray(row['tokenized_prompt_mask'])[None])
        seed=int(observation['s2_rng_seed']);key=self.jax.random.key(seed)
        normal=np.asarray(self.normal(key,obs,num_steps=10),np.float32)
        episode=str(observation['s2_episode_id']);qid=int(observation['s2_query_id'])%1000
        if qid==0:self.histories[episode]=[]
        executed=self.histories.setdefault(episode,[]);hist=np.zeros((10,7),np.float32);mask=np.zeros(10,bool);prev=executed[-10:]
        if prev:hist[-len(prev):]=np.asarray(prev,np.float32);mask[-len(prev):]=True
        hidden,groups=self.prefix(obs);hidden=np.asarray(hidden)[0];groups=np.asarray(groups)[0]
        if self.mode=='D5_HS_top1':
            feature=np.concatenate([hist.reshape(1,-1),mask.astype(np.float32).reshape(1,-1),np.asarray(obs.state,np.float32)[:,:8]],axis=1)
            with self.torch.no_grad():
                q=self.ranker.query(self.torch.as_tensor(feature,device=self.device,dtype=self.torch.float32))
                scores=self.torch.einsum('qkd,nkd->qn',q,self.candidate_keys).div(q.shape[1])*self.logit_scale
        else:
            valid=groups>=0;pooled=hidden[valid].mean(0,dtype=np.float32)[None]
            part={'pooled':pooled,'tokens':hidden[None],'groups':groups[None],'history':hist[None],'history_mask':mask[None],
                  'state':np.asarray(obs.state,np.float32)[0,:8][None]}
            with self.torch.no_grad():
                q=self.key2._model_batch_encode(self.ranker,self.ranker_mode,part,self.device)
                scores=(self.torch.einsum('qkd,nkd->qn',q,self.candidate_reps)/q.shape[1])*self.logit_scale
        ids=self.torch.argsort(scores[0],descending=True,stable=True)[:1].cpu().numpy();unit=int(ids[0])
        arm=self.units[unit,:,:6].copy();draft=np.zeros((1,10,32),np.float32);draft[0,:,:6]=arm;draft[0,:,6]=normal[0,:,6]
        score_np=scores.detach().cpu().numpy().astype(np.float32)
        feat=self.ex2.confidence_features(draft[:,:,:7],normal,score_np).reshape(1,4)
        # Accept both batched (shape [1]) and scalar outputs. The former
        # pre-release confidence checkpoints may squeeze their batch axis.
        errorhat=self.ex2.single_float(self.ex2.predict_error(self.reg,feat),name='predicted draft error')
        high_conf=self.threshold is None or errorhat<=float(self.threshold)
        low=self.t0_low if high_conf else .05
        t0=self.ex2.single_float(self.ex2.select_t0(errorhat,self.draft_scale,low=low,high=self.t0_high),name='bridge t0')
        fallback=self.threshold is not None and errorhat>float(self.threshold)
        if fallback:normalized=normal
        else:
            noise=self.jax.random.normal(self.jax.random.fold_in(key,7),(1,10,32))
            normalized=np.asarray(self.bridge(self.jax.random.key(0),obs,self.jnp.asarray(draft),self.jnp.asarray([t0]),num_steps=10,noise=noise),np.float32)
        normalized=self.ex1.restore_pi05_gripper(normalized,normal);executed.extend(normalized[0,:5,:7].copy())
        physical=self.unnormalize({'actions':normalized[0],'state':row['state']})['actions'][:,:7]
        if not np.isfinite(physical).all():raise FloatingPointError('nonfinite EX2 action output')
        return {'actions':physical.astype(np.float32),'s2_debug':{'mode':self.mode,'unit_id':unit,'unit_task':int(self.unit_task[unit]),
            'unit_source':self.unit_source[unit].astype(int).tolist(),'confidence_error':errorhat,'t0':t0,'fallback_pi05':bool(fallback),
            'history_valid_steps':int(mask.sum()),'draft_action_mse_vs_pi05':float(np.square(draft[0,:,:6]-normal[0,:,:6]).mean()),
            'seconds':time.monotonic()-started}}


def main():
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',required=True,choices=('D0_P0_top1','D1_P5_top1','D3_E_VLM_H_top1','D5_HS_top1'))
    p.add_argument('--candidate-cache',type=Path,required=True);p.add_argument('--retrieval-cache',type=Path,required=True);p.add_argument('--schedule',type=Path,required=True)
    p.add_argument('--confidence',type=Path,required=True);p.add_argument('--pi05-checkpoint',type=Path,required=True);p.add_argument('--expert-checkpoint',type=Path,required=True)
    p.add_argument('--p5-checkpoint',type=Path,required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args()
    policy=EX2Policy(mode=a.mode,candidate_cache=a.candidate_cache,retrieval_cache=a.retrieval_cache,schedule=a.schedule,
        confidence=a.confidence,pi05_checkpoint=a.pi05_checkpoint,expert_checkpoint=a.expert_checkpoint,p5_checkpoint=a.p5_checkpoint)
    WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata={'experiment':'EX2 stored motion + confidence schedule','mode':a.mode,'horizon':10}).serve_forever()

if __name__=='__main__':main()
