#!/usr/bin/env python3
"""KC2 websocket policy: learned coefficient retrieval and q-only gripper decoding."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import kq2cl_serve as S
import kq2cl_common as C
import kc1_run as K1
import kc2_run as K
from t0x_common import t0_from_disagreement,topk_disagreement


class KC2Policy:
    def __init__(self,*,mode,q_name,expert_checkpoint,bundle,calibration):
        if mode!='memory':raise ValueError(mode)
        old_bundle=K.ROOT/'kq2-cl-01'/'heads'/f'bundle_{q_name}.pt'
        self.impl=S.KQ2CLPolicy(mode='memory',query=q_name,expert_checkpoint=Path(expert_checkpoint),
            bundle=old_bundle,tpol={'const':.15,'b':None})
        p=self.impl;self.torch=p.torch;self.jax=p.jax;self.jnp=p.jnp;self.device=p.device
        result=json.loads((K.OUT/'offline_results.json').read_text());self.selected=result['selected']
        self.model,_=__import__('kc2_eval').model_from(self.selected,device=self.device)
        bank=K1.memory_bank();ck=self.torch.load(self.selected['checkpoint'],map_location='cpu',weights_only=False)
        cfg=ck['config'];self.keys_np,_,_=K1.coefficient_keys(np.asarray(bank['coefficients']),standardize=True,
            mean=np.asarray(cfg['mean']),std=np.asarray(cfg['std']))
        self.keys=self.torch.as_tensor(self.keys_np,device=self.device,dtype=self.torch.float32)
        if p.units.shape[0]!=len(self.keys):raise AssertionError('saved bank and KC2 coefficient keys differ')
        self.q_name=cfg['q_name'];self.query_vlm=self.q_name=='iii';self.threshold=float(self.selected['dev_threshold'])
        self.cal=json.loads(Path(calibration).read_text());self.D_b=float(self.cal['D_b'])
        if self.query_vlm!=(p.vlm is not None):raise AssertionError('KC2 VLM setting does not match online extractor')

    def infer(self,observation):
        p=self.impl;torch,jax,jnp=self.torch,self.jax,self.jnp;started=time.monotonic()
        row,obs=p._obs(observation);rng_key=jax.random.key(int(observation['s2_rng_seed']))
        (x,valid),sn,an,step_t=p.tokens(observation)
        vlm=None
        if self.query_vlm:
            f={k:np.asarray(v) for k,v in p.vlm(obs).items()}
            vlm=np.zeros((1,34,2048),np.float16);vlm[0,:32]=f['vlm_img'][0]
            vlm[0,32]=f['vlm_txt_mean'][0].astype(np.float16);vlm[0,33]=f['vlm_txt_last'][0].astype(np.float16)
        T=lambda a,dt=torch.float32:torch.as_tensor(a,device=self.device,dtype=dt)
        split=C.InlineSplit(T(x[None]),T(valid[None],torch.bool),None if vlm is None else T(vlm,torch.float16))
        with torch.no_grad():
            q=self.model(split,T([0],torch.long));scores=self.model.score(q,self.keys).float().cpu().numpy()
            prob_close=torch.sigmoid(self.model.grip_logits(q)).cpu().numpy()
        from kq2cl_common import summarize_scores,decode_gripper
        fake={'task':np.asarray([-1]),'episode':np.asarray([-1])}
        allowed=K1.candidate_mask(fake['task'],fake['episode'],np.asarray(p.unit_task),np.full(len(p.unit_task),-2))
        summ=summarize_scores(scores,allowed,k=10);ids=np.asarray(summ['ids'][0],np.int64);logits=np.asarray(summ['logits'][0],np.float64)
        d,_=topk_disagreement(np.asarray(p.units[ids][None],np.float32),logits[None],k=10);D=float(d[0])
        fallback=D>self.D_b;t0=float(t0_from_disagreement(D,self.cal))
        last_valid=len(an)>0;last=float(an[-1,6]) if last_valid else C.DEFAULT_GRIP
        last_sign=1. if last_valid and last>0 else -1.
        decoded=decode_gripper(prob_close,np.asarray([last_sign]),self.threshold)[0]
        unit=int(ids[0]);draft=np.zeros((1,10,32),np.float32);draft[0,:,:6]=p.units[unit,:,:6];draft[0,:,6]=decoded
        if fallback:
            output=np.asarray(p.normal(rng_key,obs,num_steps=10),np.float32).copy()
        else:
            noise=jax.random.normal(jax.random.fold_in(rng_key,7),(1,10,32))
            output=np.asarray(p.bridge(jax.random.key(0),obs,jnp.asarray(draft),jnp.asarray([t0]),num_steps=10,noise=noise),np.float32).copy()
        physical=p.unnormalize({'actions':output[0],'state':row['state']})['actions'][:,:7]
        if not np.isfinite(physical).all():raise FloatingPointError('nonfinite KC2 action')
        return {'actions':physical.astype(np.float32),'s2_debug':{'runtime':'kc2','q_name':self.q_name,
            'score_mode':self.selected['score_mode'],'lambda':self.selected['lambda'],'unit_id':unit,
            'top10_unit_ids':ids.tolist(),'disagreement_D':D,'D_b':self.D_b,'t0':t0,
            'fallback_pi05':bool(fallback),'gripper_prob_close':prob_close[0].astype(float).tolist(),
            'decoded_gripper':decoded.astype(float).tolist(),'step_t':step_t,
            'history_valid_steps':int(valid.sum()),'seconds':time.monotonic()-started}}


def main():
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('memory',),required=True)
    ap.add_argument('--query',choices=('i','iii'),required=True);ap.add_argument('--expert-checkpoint',required=True)
    ap.add_argument('--bundle',required=True);ap.add_argument('--tpol',default='{}');ap.add_argument('--calibration')
    ap.add_argument('--port',type=int,required=True)
    a=ap.parse_args();tp=json.loads(a.tpol);cal=a.calibration or tp.get('calibration')
    if not cal:raise ValueError('KC2 calibration path is required')
    policy=KC2Policy(mode=a.mode,q_name=a.query,expert_checkpoint=a.expert_checkpoint,bundle=a.bundle,calibration=cal)
    WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata={'experiment':'KC2','query':policy.q_name,'horizon':10}).serve_forever()

if __name__=='__main__':main()
