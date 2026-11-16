"""Websocket policy adapter for SEG2's selected segment-context model."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PI05=HERE.parent/'pi05_memory'
if str(PI05) not in sys.path:sys.path.insert(0,str(PI05))
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))


def make_policy(out,checkpoint,*,variant='selected'):
    import torch
    from openpi.models.model import Observation
    from serve import MemoryPolicy
    import seg_context_retrieval as sc
    import seg_context_run as runner
    class Policy(MemoryPolicy):
        def __init__(self):
            super().__init__(Path(out),Path(checkpoint),normal_only=True)
            torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
            torch.cuda.set_per_process_memory_fraction(float(os.environ.get('TORCH_MEM_FRACTION','.10')))
            self.model,self.selected,self.ckpt=runner._load_model(out,4101,np.load(Path(out)/'memory_bank.npz',allow_pickle=False))
            self.model.to('cuda').eval()
            bankfile=np.load(Path(out)/'online_bank.npz',allow_pickle=False)
            self.bank={k:bankfile[k] for k in bankfile.files}
            self.bank['keys']=torch.as_tensor(np.asarray(self.bank['keys']),device='cuda',dtype=torch.float32)
            cweight=self.selected['continuity']
            if cweight=='learned':cweight=float(torch.nn.functional.softplus(self.model.encoder.learned_continuity_scale).detach().cpu())
            cfg=sc.SegmentConfig(token_dim=2048,width=256,heads=4,cross_layers=int(self.selected['layers']),
                continuity_weight=cweight,exclude_same_task=False,
                boundary_blend=False,loop_exclude_previous_unit=(variant=='exclude_previous'),
                value_head_mode=self.selected.get('value_head_mode','dct'),use_pose_scale_head=(variant!='no_pose_head'))
            self.segment_policy=sc.SegmentContextPolicy(self.model.encoder,self.bank,prefix_fn=self._prefix,
                pi05_fn=lambda obs:self._current_normal[0,:,:7],config=cfg,pose_head=self.model.pose_head)
            self.variant=variant;self._current_normal=None;self._episode=None;self._theta=None
        def _prefix(self,obs):
            h,g=self.features(obs);return np.asarray(h)[0],np.asarray(g)[0]
        def infer(self,observation):
            raw={k:observation[k] for k in ('observation/image','observation/wrist_image','observation/state','prompt')}
            row=self.transform(raw)
            obs=Observation(images={k:self.jnp.asarray(v)[None].astype(self.jnp.float32)/255*2-1 for k,v in row['image'].items()},
                image_masks={k:self.jnp.asarray([v]) for k,v in row['image_mask'].items()},state=self.jnp.asarray(row['state'])[None],
                tokenized_prompt=self.jnp.asarray(row['tokenized_prompt'])[None],tokenized_prompt_mask=self.jnp.asarray(row['tokenized_prompt_mask'])[None])
            rng=self.jax.random.key(int(observation['s2_rng_seed']))
            normal=np.asarray(self.normal(rng,obs,num_steps=10))[0].astype(np.float32)
            self._current_normal=normal[None]
            episode=observation.get('s2_episode_id','unknown');qid=int(observation.get('s2_query_id',0))
            if self._episode!=episode or qid==0:
                self.segment_policy.reset();self._episode=episode
            # Query tasks are held out from the 40-task memory. Use sentinel IDs
            # so coincident numeric task/episode IDs cannot cause false masking.
            act,meta=self.segment_policy.act(obs,np.asarray(row['state'],np.float32)[:8],task_id=-1,episode_id=-999999)
            norm=np.zeros((10,32),np.float32);norm[:,:7]=np.asarray(act,np.float32)[:10,:7]
            # The next query is five controls later; feed only those actions to
            # the causal history as the controls that were issued to the worker.
            st=np.repeat(np.asarray(row['state'],np.float32)[None,:8],5,axis=0)
            self.segment_policy.observe_executed(norm[:5,:7],st)
            raw_action=self.unnormalize({'actions':norm,'state':row['state']})['actions'][:,:7]
            if not np.isfinite(raw_action).all():raise FloatingPointError('nonfinite SEG2 action')
            return {'actions':raw_action.astype(np.float32),'s2_debug':{'condition':'seg2-'+self.variant,**meta,
                'query_id':qid,'history_steps':len(self.segment_policy.previous_actions),'seconds':time.monotonic()}}
    return Policy()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--variant',choices=('selected','no_pose_head','exclude_previous'),default='selected');p.add_argument('--port',type=int,required=True);a=p.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy=make_policy(a.output,a.checkpoint,variant=a.variant)
    WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata={'policy':'SEG2','variant':a.variant}).serve_forever()

if __name__=='__main__':main()
