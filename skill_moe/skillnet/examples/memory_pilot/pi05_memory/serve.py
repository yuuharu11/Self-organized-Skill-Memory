"""Serve both matched conditions without modifying the official fallback."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


class TokenIndex:
    def __init__(self, output, *, block=32):
        import torch
        self.torch = torch
        self.ids = np.load(output/'memory_row_ids.npy')
        tokens = np.load(output/'tokens.npy',mmap_mode='r')
        groups = np.load(output/'groups.npy',mmap_mode='r')
        self.keys = torch.as_tensor(np.asarray(tokens[self.ids]),device='cuda',dtype=torch.float16)
        self.keys /= torch.linalg.vector_norm(self.keys.float(),dim=-1,keepdim=True).clamp_min(1e-8).half()
        self.groups = torch.as_tensor(np.asarray(groups[self.ids]),device='cuda')
        text = np.flatnonzero(np.any(np.asarray(groups[self.ids])[:,512:]==2,axis=0))
        self.ranges = [(0,256),(256,512),(512,512+int(text[-1])+1)]
        self.block = block

    def search(self, query, groups):
        torch = self.torch
        with torch.inference_mode():
            q = torch.as_tensor(np.asarray(query).copy(),device='cuda').float()
            q = (q/torch.linalg.vector_norm(q,dim=-1,keepdim=True).clamp_min(1e-8)).half()
            scores = torch.empty(len(self.ids),device='cuda')
            for start in range(0,len(self.ids),self.block):
                end=min(start+self.block,len(self.ids))
                terms=[]
                for group,(left,right) in enumerate(self.ranges):
                    selected=q[np.asarray(groups)==group]
                    if not len(selected):
                        raise ValueError('query is missing a required modality')
                    sim=torch.matmul(selected[None],self.keys[start:end,left:right].transpose(1,2)).float()
                    sim.masked_fill_(self.groups[start:end,None,left:right]!=group,-torch.inf)
                    terms.append(sim.amax(-1).mean(-1))
                scores[start:end]=torch.stack(terms).mean(0)
            best=int(scores.argmax().item())
            return int(self.ids[best]),float(scores[best])


class MemoryPolicy:
    def __init__(self, output, checkpoint, *, normal_only=False):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        import orbax.checkpoint as ocp
        from openpi.shared.nnx_utils import module_jit,PathRegex
        from openpi import transforms
        from openpi.shared.normalize import load
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.policies.libero_policy import LiberoInputs
        from openpi.models.model import ModelType
        from run import load_policy,restore_codec
        self.jax,self.jnp=jax,jnp
        self.output=output
        self.model,_=load_policy(checkpoint)
        self.normal=module_jit(self.model.sample_actions,static_argnames=('num_steps',))
        self.features=module_jit(self.model.prefix_features)
        self.normal_only=normal_only
        if not normal_only:
            filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
            selection=output/'best_bridge.json'
            if not selection.exists():selection=output/'latest_bridge.json'
            path=Path(json.loads(selection.read_text())['path'])
            with ocp.PyTreeCheckpointer() as loader:
                partial=loader.restore(path)
            state=nnx.state(self.model,filt)
            state.replace_by_pure_dict(jax.tree.map(jnp.asarray,partial));nnx.update(self.model,state)
            self.bridge=module_jit(self.model.sample_bridge,static_argnames=('num_steps',))
            self.index=TokenIndex(output)
            self.codec=restore_codec(output)
            self.bases=np.load(output/'bases.npy',mmap_mode='r')
            self.schedule=json.loads((output/'calibration.json').read_text())['schedule']
        normfile=list((checkpoint/'assets').rglob('norm_stats.json'))
        if len(normfile)!=1:
            raise ValueError('ambiguous normalization')
        stats=load(normfile[0].parent)
        self.transform=transforms.compose([
            LiberoInputs(ModelType.PI05),transforms.Normalize(stats,use_quantiles=True),
            transforms.ResizeImages(224,224),
            transforms.TokenizePrompt(PaligemmaTokenizer(200),discrete_state_input=False),
            transforms.PadStatesAndActions(32)])
        self.unnormalize=transforms.Unnormalize(stats,use_quantiles=True)

    def infer(self, observation):
        from openpi.models.model import Observation
        from calibrate import apply_schedule
        start=time.monotonic()
        # Explicit whitelist: SkillNet annotations from the legacy client never
        # enter this dense policy, its prompt, or its search query.
        raw={k:observation[k] for k in ['observation/image','observation/wrist_image','observation/state','prompt']}
        row=self.transform(raw)
        obs=Observation(
            images={k:self.jnp.asarray(v)[None].astype(self.jnp.float32)/255*2-1 for k,v in row['image'].items()},
            image_masks={k:self.jnp.asarray([v]) for k,v in row['image_mask'].items()},
            state=self.jnp.asarray(row['state'])[None],
            tokenized_prompt=self.jnp.asarray(row['tokenized_prompt'])[None],
            tokenized_prompt_mask=self.jnp.asarray(row['tokenized_prompt_mask'])[None])
        key=self.jax.random.key(int(observation['s2_rng_seed']))
        condition=observation['s2_condition']
        debug={'condition':condition,'fallback':True}
        if condition=='normal':
            result=self.normal(key,obs,num_steps=10)
        elif condition=='memory':
            if self.normal_only:raise ValueError('memory condition unavailable in normal-only smoke')
            hidden,groups=self.features(obs)
            hidden,groups=np.asarray(hidden)[0],np.asarray(groups)[0]
            memory_row,score=self.index.search(hidden,groups)
            mode=int(apply_schedule(np.asarray([score]),self.schedule)[0])
            debug.update(memory_row=memory_row,score=score,mode=mode,fallback=mode==0)
            if mode==0:
                result=self.normal(key,obs,num_steps=10)
            else:
                feature=hidden[groups>=0].mean(0)[None]
                state=np.asarray(obs.state)
                _,residual=self.codec.prior(feature,state)
                draft=self.codec.decode(np.asarray(self.bases[memory_row])[None],residual,feature,state)
                t0=[1.,.6,.4,.2][mode]
                debug['t0']=t0
                result=self.bridge(key,obs,self.jnp.asarray(draft),self.jnp.asarray([t0]),num_steps=10)
        else:
            raise ValueError(f'unknown experimental condition: {condition}')
        actions=np.asarray(result)[0]
        actions=self.unnormalize({'actions':actions,'state':row['state']})['actions'][:,:7]
        if not np.isfinite(actions).all():
            raise FloatingPointError('nonfinite policy action')
        debug['seconds']=time.monotonic()-start
        return {'actions':actions.astype(np.float32),'s2_debug':debug}


def main():
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8154)
    parser.add_argument('--normal-only',action='store_true')
    args=parser.parse_args()
    policy=MemoryPolicy(args.output,args.checkpoint,normal_only=args.normal_only)
    WebsocketPolicyServer(policy,host='127.0.0.1',port=args.port,
        metadata={'experiment':'pi05_memory40','conditions':['normal','memory'],'action_horizon':10}).serve_forever()


if __name__=='__main__':
    main()
