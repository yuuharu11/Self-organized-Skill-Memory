#!/usr/bin/env python3
"""XF2 online policy: KC2 retrieval/Gdec with isotonic confidence and V/F t0."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import kc2_serve as S
import kq2cl_common as C
import kc2_run as K
import kc1_run as K1


class XF2Policy:
    def __init__(self,*,mode,query,expert_checkpoint,bundle,tpol):
        if query!='i':raise ValueError('XF2 uses the KC2 selected query (i)')
        self.impl=S.KC2Policy(mode=mode,q_name=query,expert_checkpoint=Path(expert_checkpoint),
                             bundle=Path(bundle),calibration=Path(tpol['calibration']))
        p=self.impl
        self.torch=p.torch;self.jax=p.jax;self.jnp=p.jnp;self.device=p.device
        self.strategy=str(tpol['strategy'])
        if self.strategy not in ('XF2-V','XF2-F'):raise ValueError(self.strategy)
        self.cal=json.loads(Path(tpol['calibration']).read_text())
        iso=self.cal['isotonic'];self.iso_x=np.asarray(iso['x'],np.float64);self.iso_y=np.asarray(iso['y'],np.float64)
        self.fallback_threshold=float(self.cal['fallback']['threshold'])
        offline=json.loads((Path(tpol['calibration']).parent/'offline_results.json').read_text())
        self.k=float(offline['selected_settings']['variable_k']);self.fixed=float(offline['selected_settings']['fixed_t0'])
        self.gdec_threshold=float(self.impl.threshold)

    def infer(self,observation):
        p=self.impl;torch,jax,jnp=self.torch,self.jax,self.jnp;started=time.monotonic()
        row,obs=p.impl._obs(observation);rng_key=jax.random.key(int(observation['s2_rng_seed']))
        (x,valid),sn,an,step_t=p.impl.tokens(observation)
        T=lambda a,dt=torch.float32:torch.as_tensor(a,device=self.device,dtype=dt)
        split=C.InlineSplit(T(x[None]),T(valid[None],torch.bool),None)
        with torch.no_grad():
            q=p.model(split,T([0],torch.long));scores=p.model.score(q,p.keys).float().cpu().numpy()
            probs=torch.sigmoid(p.model.grip_logits(q)).float().cpu().numpy()
        summ=C.summarize_scores(scores,np.ones_like(scores,dtype=bool),k=10)
        ids=np.asarray(summ['ids'][0],np.int64);logits=np.asarray(summ['logits'][0],np.float64)
        key=np.asarray(p.keys_np[ids],np.float32)
        spread=float(np.var(key,axis=0).mean())
        ehat=float(np.interp(spread,self.iso_x,self.iso_y))
        if self.strategy=='XF2-V':
            root=np.sqrt(max(ehat,0.));t0=float(np.clip(self.k*root/(1+self.k*root),.05,.6))
        else:t0=self.fixed
        fb=self.cal['fallback'];tie=fb.get('spread_tiebreak_threshold')
        fallback=ehat>self.fallback_threshold or (tie is not None and ehat==self.fallback_threshold and spread>=float(tie))
        last_valid=len(an)>0;last=float(an[-1,6]) if last_valid else C.DEFAULT_GRIP
        last_sign=1. if last_valid and last>0 else -1.
        grip=C.decode_gripper(probs,np.asarray([last_sign]),self.gdec_threshold)[0]
        unit=int(ids[0]);draft=np.zeros((1,10,32),np.float32);draft[0,:,:6]=p.impl.units[unit,:,:6];draft[0,:,6]=grip
        if fallback:
            output=np.asarray(p.impl.normal(rng_key,obs,num_steps=10),np.float32).copy()
        else:
            noise=jax.random.normal(jax.random.fold_in(rng_key,7),(1,10,32))
            output=np.asarray(p.impl.bridge(jax.random.key(0),obs,jnp.asarray(draft),jnp.asarray([t0]),num_steps=10,noise=noise),np.float32).copy()
        physical=p.impl.unnormalize({'actions':output[0],'state':row['state']})['actions'][:,:7]
        if not np.isfinite(physical).all():raise FloatingPointError('nonfinite XF2 action')
        return {'actions':physical.astype(np.float32),'s2_debug':{'runtime':'xf2','strategy':self.strategy,
          'unit_id':unit,'top10_unit_ids':ids.tolist(),'spread10':spread,'confidence_error':ehat,
          'fallback_threshold':self.fallback_threshold,'fallback_pi05':bool(fallback),'t0':t0,
          'gripper_prob_close':probs[0].astype(float).tolist(),'decoded_gripper':grip.astype(float).tolist(),
          'step_t':step_t,'history_valid_steps':int(valid.sum()),'seconds':time.monotonic()-started}}


def main():
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('memory',),required=True)
    ap.add_argument('--query',choices=('i',),required=True);ap.add_argument('--expert-checkpoint',required=True)
    ap.add_argument('--bundle',required=True);ap.add_argument('--tpol',required=True);ap.add_argument('--port',type=int,required=True)
    a=ap.parse_args();pol=XF2Policy(mode=a.mode,query=a.query,expert_checkpoint=a.expert_checkpoint,
        bundle=a.bundle,tpol=json.loads(a.tpol))
    WebsocketPolicyServer(pol,host='127.0.0.1',port=a.port,metadata={'experiment':'XF2','strategy':pol.strategy,'horizon':10}).serve_forever()

if __name__=='__main__':main()
