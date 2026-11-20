#!/usr/bin/env python3
"""UQ1 closed-loop server using fixed GRU×VLM query and XF2-style Expert t0."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import kc2_serve as KC2S
import kq2cl_common as C
import kc2_run as K2
import kc1_run as K1
import xf2_run as X
from uq1_run import UQ1Net, spread10

class UQ1Policy:
    def __init__(self,*,mode,query,expert_checkpoint,bundle,tpol):
        if query!='iii':raise ValueError('UQ1 is fixed to query iii')
        # KC2's selected checkpoint is query i, so constructing its wrapper
        # incorrectly asserts that the online extractor has no VLM.  Build the
        # shared pi05 observation/expert runtime directly with query iii; UQ1
        # supplies its own retrieval and grip heads below.
        old_bundle=K2.ROOT/'kq2-cl-01'/'heads'/'bundle_iii.pt'
        self.inner=KC2S.S.KQ2CLPolicy(mode=mode,query='iii',expert_checkpoint=Path(expert_checkpoint),
             bundle=old_bundle,tpol={'const':.15,'b':None})
        self.torch=self.inner.torch;self.jax=self.inner.jax;self.jnp=self.inner.jnp;self.device=self.inner.device
        bank=K1.memory_bank();_,_,self.keys_np,_,_=X._kc2_assets(bank)
        self.keys=self.torch.as_tensor(self.keys_np,device=self.device,dtype=self.torch.float32)
        self.units=np.asarray(bank['value_actions10'],np.float32)
        if len(self.units)!=len(self.keys_np):raise AssertionError('UQ1 coefficient keys/action bank mismatch')
        blob=self.torch.load(Path(tpol['query_checkpoint']),map_location='cpu',weights_only=False)
        self.model=UQ1Net().to(self.device);self.model.load_state_dict(blob['state_dict']);self.model.eval()
        self.threshold=float(tpol['gripper_threshold']);self.strategy=str(tpol['strategy'])
        self.cal=json.loads(Path(tpol['calibration']).read_text());self.iso_x=np.asarray(self.cal['isotonic']['x'],float);self.iso_y=np.asarray(self.cal['isotonic']['y'],float)
        self.fb=self.cal['fallback'];offline=json.loads((Path(tpol['calibration']).parent/'offline_results.json').read_text())
        self.k=float(offline['selected_settings']['variable_k']);self.fixed=float(offline['selected_settings']['fixed_t0'])

    def infer(self,observation):
        p=self.inner;torch,jax,jnp=self.torch,self.jax,self.jnp;started=time.monotonic()
        row,obs=p._obs(observation);rng_key=jax.random.key(int(observation['s2_rng_seed']))
        (x,valid),sn,an,step_t=p.tokens(observation)
        feat={k:np.asarray(v) for k,v in p.vlm(obs).items()}
        vlm=np.zeros((1,34,2048),np.float16);vlm[0,:32]=feat['vlm_img'][0];vlm[0,32]=feat['vlm_txt_mean'][0].astype(np.float16);vlm[0,33]=feat['vlm_txt_last'][0].astype(np.float16)
        T=lambda a,dt=torch.float32:torch.as_tensor(a,device=self.device,dtype=dt)
        split=C.InlineSplit(T(x[None]),T(valid[None],torch.bool),T(vlm,torch.float16))
        with torch.inference_mode():
            q=self.model(split,T([0],torch.long));scores=self.model.score(q,self.keys).float().cpu().numpy()
            prob=torch.sigmoid(self.model.grip_logits(q)).float().cpu().numpy()
        sm=C.summarize_scores(scores,np.ones_like(scores,dtype=bool),k=10);ids=np.asarray(sm['ids'][0],np.int64)
        spread=float(spread10(self.keys_np,ids[None])[0]);ehat=float(np.interp(spread,self.iso_x,self.iso_y))
        t0=float(np.clip(self.k*np.sqrt(max(ehat,0))/(1+self.k*np.sqrt(max(ehat,0))),.05,.6)) if self.strategy=='UQ1-V' else self.fixed
        tie=self.fb.get('spread_tiebreak_threshold');fallback=ehat>float(self.fb['threshold']) or (tie is not None and ehat==float(self.fb['threshold']) and spread>=float(tie))
        last=float(an[-1,6]) if len(an) else C.DEFAULT_GRIP;lastsign=1. if last>0 else -1.
        grip=C.decode_gripper(prob,np.asarray([lastsign]),self.threshold)[0]
        unit=int(ids[0]);draft=np.zeros((1,10,32),np.float32);draft[0,:,:6]=p.units[unit,:,:6];draft[0,:,6]=grip
        if fallback:out=np.asarray(p.normal(rng_key,obs,num_steps=10),np.float32).copy()
        else:
            noise=jax.random.normal(jax.random.fold_in(rng_key,7),(1,10,32))
            out=np.asarray(p.bridge(jax.random.key(0),obs,jnp.asarray(draft),jnp.asarray([t0]),num_steps=10,noise=noise),np.float32).copy()
        physical=p.unnormalize({'actions':out[0],'state':row['state']})['actions'][:,:7]
        if not np.isfinite(physical).all():raise FloatingPointError('nonfinite UQ1 action')
        return {'actions':physical.astype(np.float32),'s2_debug':{'runtime':'uq1','strategy':self.strategy,'unit_id':unit,
           'top10_unit_ids':ids.tolist(),'spread10':spread,'confidence_error':ehat,'fallback_pi05':bool(fallback),'t0':t0,
           'gripper_threshold':self.threshold,'step_t':step_t,'seconds':time.monotonic()-started}}

def main():
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('memory',),required=True);ap.add_argument('--query',choices=('iii',),required=True)
    ap.add_argument('--expert-checkpoint',required=True);ap.add_argument('--bundle',required=True);ap.add_argument('--tpol',required=True);ap.add_argument('--port',type=int,required=True)
    a=ap.parse_args();pol=UQ1Policy(mode=a.mode,query=a.query,expert_checkpoint=a.expert_checkpoint,bundle=a.bundle,tpol=json.loads(a.tpol))
    WebsocketPolicyServer(pol,host='127.0.0.1',port=a.port,metadata={'experiment':'UQ1','query':'iii','strategy':pol.strategy,'horizon':10}).serve_forever()
if __name__=='__main__':main()
