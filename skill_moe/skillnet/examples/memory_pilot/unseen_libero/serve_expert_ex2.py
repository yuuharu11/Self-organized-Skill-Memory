#!/usr/bin/env python3
"""Runtime server for EX2 motion-only Expert with confidence-conditioned t0."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;PI05=HERE.parent/'pi05_memory'
REPO_ROOT = Path(__file__).resolve().parents[5]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
sys.path[:0]=[str(HERE),str(PI05)]


class EX2Policy:
    def __init__(self, *, mode, candidate_cache, retrieval_cache, schedule, confidence,
                 pi05_checkpoint, expert_checkpoint, p5_checkpoint, kq_query_checkpoint=None,
                 kq_keys=None, use_expert=True, ekq_bundle=None, ekq_gripper='head'):
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
        if mode not in MODES:raise ValueError(mode)
        # CL1: KQ_* modes replace only the unit ranker by a KQ-final query
        # (key_final.py) over fixed K-x1 motion keys; use_expert=False replays
        # the retrieved arm draft directly (gripper still from π0.5).
        self.use_expert=bool(use_expert)
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
            if mode in KQ_MODES:
                import key_final as kf
                blob=torch.load(kq_query_checkpoint,map_location='cpu',weights_only=False);self.kq_cfg=blob['cfg']
                expected={'KQ_AFIX_top1':'mqe','KQ_HMLP_top1':'hsmlp'}[mode]
                if self.kq_cfg['query']!=expected or self.kq_cfg['input']!='e_h' or self.kq_cfg['key']!='x1_mk1b':
                    raise ValueError(f'{mode} expects an e_h {expected} query over fixed x1_mk1b keys, got {self.kq_cfg}')
                self.ranker=kf.make_query(self.kq_cfg).to(self.device);self.ranker.load_state_dict(blob['query']);self.ranker.eval()
                self.kq_score=kf.score
                self.candidate_keys=torch.as_tensor(np.load(kq_keys),device=self.device,dtype=torch.float32)
                if tuple(self.candidate_keys.shape)!=(len(self.units),256):raise AssertionError(f'K-x1 key cache shape {tuple(self.candidate_keys.shape)}')
            elif mode=='D5_HS_top1':
                d5=json.loads(((STATE_DIR / 'outputs/memory-pilot/unseen-libero/key-grid-01/production-04/results.json')).read_text())['new_models']['history_state']['variants']['two_tower_mlp']
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
        # EKQ (expert_kq_runtime.py): π0.5-free confidence/t0 and gripper head for KQ_* modes.
        # π0.5 is sampled only for the Dev-chosen fallback or the π0.5-gripper ablation.
        self.ekq=None
        if ekq_bundle is not None:
            if mode not in KQ_MODES:raise ValueError('--ekq-bundle requires a KQ_* mode')
            if ekq_gripper not in ('head','pi05'):raise ValueError(ekq_gripper)
            import expert_kq_runtime as ekq
            b=torch.load(ekq_bundle,map_location='cpu',weights_only=False)
            cm=ekq.new_conf_head(b['confidence']['kind']);cm.load_state_dict(b['confidence']['state_dict']);cm.eval()
            gm=ekq.new_grip_head(b['gripper']['hidden']);gm.load_state_dict(b['gripper']['state_dict']);gm.eval()
            self.ekq={'mod':ekq,'conf':cm,'grip':gm,'map':b['map'],'gripper':ekq_gripper}

    def infer(self,observation):
        started=time.monotonic();raw={k:observation[k] for k in ('observation/image','observation/wrist_image','observation/state','prompt')}
        row=self.transform(raw)
        obs=self.Observation(images={k:self.jnp.asarray(v)[None].astype(self.jnp.float32)/255*2-1 for k,v in row['image'].items()},
            image_masks={k:self.jnp.asarray([v]) for k,v in row['image_mask'].items()},state=self.jnp.asarray(row['state'])[None],
            tokenized_prompt=self.jnp.asarray(row['tokenized_prompt'])[None],tokenized_prompt_mask=self.jnp.asarray(row['tokenized_prompt_mask'])[None])
        seed=int(observation['s2_rng_seed']);key=self.jax.random.key(seed)
        if self.ekq is not None:return self._infer_ekq(observation,row,obs,key,started)
        normal=np.asarray(self.normal(key,obs,num_steps=10),np.float32)
        episode=str(observation['s2_episode_id']);qid=int(observation['s2_query_id'])%1000
        if qid==0:self.histories[episode]=[]
        executed=self.histories.setdefault(episode,[]);hist=np.zeros((10,7),np.float32);mask=np.zeros(10,bool);prev=executed[-10:]
        if prev:hist[-len(prev):]=np.asarray(prev,np.float32);mask[-len(prev):]=True
        hidden,groups=self.prefix(obs);hidden=np.asarray(hidden)[0];groups=np.asarray(groups)[0]
        if self.mode in KQ_MODES:
            st=np.asarray(obs.state,np.float32)[:,:8]
            feature=np.concatenate([hist.reshape(1,-1),mask.astype(np.float32).reshape(1,-1),st],axis=1)
            T=lambda a,dt=self.torch.float32:self.torch.as_tensor(a,device=self.device,dtype=dt)
            with self.torch.no_grad():
                q=self.ranker(history=T(hist[None]),history_mask=T(mask[None],self.torch.bool),state=T(st),hs=T(feature))
                scores=self.kq_score(q,self.candidate_keys,'dot',self.ranker.logit_scale)
        elif self.mode=='D5_HS_top1':
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
        if not self.use_expert:normalized=draft.copy()
        elif fallback:normalized=normal
        else:
            noise=self.jax.random.normal(self.jax.random.fold_in(key,7),(1,10,32))
            normalized=np.asarray(self.bridge(self.jax.random.key(0),obs,self.jnp.asarray(draft),self.jnp.asarray([t0]),num_steps=10,noise=noise),np.float32)
        normalized=self.ex1.restore_pi05_gripper(normalized,normal);executed.extend(normalized[0,:5,:7].copy())
        physical=self.unnormalize({'actions':normalized[0],'state':row['state']})['actions'][:,:7]
        if not np.isfinite(physical).all():raise FloatingPointError('nonfinite EX2 action output')
        return {'actions':physical.astype(np.float32),'s2_debug':{'mode':self.mode,'unit_id':unit,'unit_task':int(self.unit_task[unit]),
            'unit_source':self.unit_source[unit].astype(int).tolist(),'confidence_error':errorhat,'t0':t0,'fallback_pi05':bool(fallback),
            'history_valid_steps':int(mask.sum()),'expert_applied':bool(self.use_expert and not fallback),
            'top1_score':float(score_np[0,unit]),
            **({'query_feature':feature[0].astype(float).tolist()} if os.environ.get('CL1_DEBUG_FEATURES')=='1' and self.mode in KQ_MODES+('D5_HS_top1',) else {}),'score_margin12':float(np.diff(np.sort(score_np[0])[-2:])[0]),'draft_action_mse_vs_pi05':float(np.square(draft[0,:,:6]-normal[0,:,:6]).mean()),
            'seconds':time.monotonic()-started}}

    def _infer_ekq(self,observation,row,obs,key,started):
        E=self.ekq;mod=E['mod']
        episode=str(observation['s2_episode_id']);qid=int(observation['s2_query_id'])%1000
        if qid==0:self.histories[episode]=[]
        executed=self.histories.setdefault(episode,[]);hist=np.zeros((10,7),np.float32);mask=np.zeros(10,bool);prev=executed[-10:]
        if prev:hist[-len(prev):]=np.asarray(prev,np.float32);mask[-len(prev):]=True
        st=np.asarray(obs.state,np.float32)[:,:8]
        feature=np.concatenate([hist.reshape(1,-1),mask.astype(np.float32).reshape(1,-1),st],axis=1)
        T=lambda a,dt=self.torch.float32:self.torch.as_tensor(a,device=self.device,dtype=dt)
        with self.torch.no_grad():
            q=self.ranker(history=T(hist[None]),history_mask=T(mask[None],self.torch.bool),state=T(st),hs=T(feature))
            scores=self.kq_score(q,self.candidate_keys,'dot',self.ranker.logit_scale)
        score_np=scores.double().cpu().numpy();qnorm=q.float().norm(dim=-1).cpu().numpy()
        summ=mod.score_summary(score_np);unit=int(summ['ids'][0,0])
        feats=mod.confidence_features(summ['ids'],summ['vals'],summ['entropy'],summ['top1_prob'],self.units,hist[None],mask[None],st,qnorm)
        errorhat=float(mod.predict_conf(E['conf'],feats)[0]);m=E['map']
        t0=float(mod.select_t0(errorhat,m['c'],m['lo'],m['hi']))
        fallback=m['threshold'] is not None and errorhat>float(m['threshold'])
        need_normal=fallback or E['gripper']=='pi05'
        normal=np.asarray(self.normal(key,obs,num_steps=10),np.float32) if need_normal else None
        gin=mod.gripper_inputs(summ['ids'],summ['vals'],self.units,hist[None],mask[None],st)
        grip=mod.predict_grip(E['grip'],gin,hist[None],mask[None])[0]
        gcol=normal[0,:,6] if E['gripper']=='pi05' else grip
        draft=np.zeros((1,10,32),np.float32);draft[0,:,:6]=self.units[unit,:,:6];draft[0,:,6]=gcol
        if fallback:normalized=normal.copy()
        elif not self.use_expert:normalized=draft.copy()
        else:
            noise=self.jax.random.normal(self.jax.random.fold_in(key,7),(1,10,32))
            normalized=np.asarray(self.bridge(self.jax.random.key(0),obs,self.jnp.asarray(draft),self.jnp.asarray([t0]),num_steps=10,noise=noise),np.float32).copy()
            normalized[0,:,6]=gcol
        executed.extend(normalized[0,:5,:7].copy())
        physical=self.unnormalize({'actions':normalized[0],'state':row['state']})['actions'][:,:7]
        if not np.isfinite(physical).all():raise FloatingPointError('nonfinite EKQ action output')
        return {'actions':physical.astype(np.float32),'s2_debug':{'mode':self.mode,'runtime':'ekq','gripper_source':E['gripper'],'unit_id':unit,
            'unit_task':int(self.unit_task[unit]),'unit_source':self.unit_source[unit].astype(int).tolist(),'confidence_error':errorhat,'t0':t0,
            'fallback_pi05':bool(fallback),'pi05_sampled':bool(need_normal),'history_valid_steps':int(mask.sum()),
            'expert_applied':bool(self.use_expert and not fallback),'top1_score':float(score_np[0,unit]),
            'gripper_chunk':[float(x) for x in normalized[0,:,6]],
            **({'query_feature':feature[0].astype(float).tolist(),'confidence_features':feats[0].astype(float).tolist()} if os.environ.get('CL1_DEBUG_FEATURES')=='1' else {}),
            'score_margin12':float(summ['vals'][0,0]-summ['vals'][0,1]),'seconds':time.monotonic()-started}}


MODES=('D0_P0_top1','D1_P5_top1','D3_E_VLM_H_top1','D5_HS_top1','KQ_AFIX_top1','KQ_HMLP_top1')
KQ_MODES=('KQ_AFIX_top1','KQ_HMLP_top1')


def main():
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',required=True,choices=MODES)
    p.add_argument('--candidate-cache',type=Path,required=True);p.add_argument('--retrieval-cache',type=Path,required=True);p.add_argument('--schedule',type=Path,required=True)
    p.add_argument('--confidence',type=Path,required=True);p.add_argument('--pi05-checkpoint',type=Path,required=True);p.add_argument('--expert-checkpoint',type=Path,required=True)
    p.add_argument('--p5-checkpoint',type=Path,required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--kq-query-checkpoint',type=Path);p.add_argument('--kq-keys',type=Path);p.add_argument('--no-expert',action='store_true')
    p.add_argument('--ekq-bundle',type=Path,help='EKQ runtime bundle (expert_kq_runtime.py bundle); KQ_* modes only')
    p.add_argument('--ekq-gripper',default='head',choices=('head','pi05'));a=p.parse_args()
    policy=EX2Policy(mode=a.mode,candidate_cache=a.candidate_cache,retrieval_cache=a.retrieval_cache,schedule=a.schedule,
        confidence=a.confidence,pi05_checkpoint=a.pi05_checkpoint,expert_checkpoint=a.expert_checkpoint,p5_checkpoint=a.p5_checkpoint,
        kq_query_checkpoint=a.kq_query_checkpoint,kq_keys=a.kq_keys,use_expert=not a.no_expert,
        ekq_bundle=a.ekq_bundle,ekq_gripper=a.ekq_gripper)
    WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata={'experiment':'EX2 stored motion + confidence schedule','mode':a.mode,'expert':not a.no_expert,'horizon':10}).serve_forever()

if __name__=='__main__':main()
