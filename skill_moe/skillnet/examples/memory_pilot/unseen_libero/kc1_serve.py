#!/usr/bin/env python3
"""KC1 online retrieval server: coefficient query, T5 disagreement, KC1 gripper decoder."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import kq2cl_serve as S
import kq2cl_common as C
import kc1_run as K
import kc1_head as H
from t0x_common import t0_from_disagreement,topk_disagreement


class KC1Policy:
    def __init__(self,*,mode,query,expert_checkpoint,bundle,tpol):
        if mode!='memory':raise ValueError('KC1 only uses memory mode')
        self.query=query;self.tpol=tpol
        old_bundle=C.ROOT/'kq2-cl-01'/'heads'/f'bundle_{query}.pt'
        # Reuse the exact KQ2-CL observation, pi0.5, history and Expert bridge path.
        self.impl=S.KQ2CLPolicy(mode='memory',query=query,expert_checkpoint=Path(expert_checkpoint),
                                bundle=old_bundle,tpol={'const':.15,'b':None})
        p=self.impl;device=p.device;torch=p.torch
        meta=json.loads((K.OUT/'offline_corrected.json').read_text());self.selected=meta['selected']
        blob=torch.load(self.selected['checkpoint'],map_location='cpu',weights_only=False)
        cfg=blob['cfg'];self.query=cfg['query'];self.standardize=bool(cfg['standardize'])
        from kq2_run import QueryNet
        self.qnet=QueryNet('gru',1,1 if self.query=='iii' else 0,self.query=='iii',history='w10',dct=True).to(device)
        self.qnet.load_state_dict(blob['query']);self.qnet.eval()
        coeff=np.load(K.BANK_DIR/'bank__coefficients.npy').astype(np.float32)
        self.keys_np,_,_=K.coefficient_keys(coeff,standardize=self.standardize,mean=np.asarray(cfg['mean']),std=np.asarray(cfg['std']))
        self.keys=torch.as_tensor(self.keys_np,device=device,dtype=torch.float32)
        if p.units.shape[0]!=len(self.keys):raise AssertionError('KC1 coeff keys and stored action units differ')
        self.scale=float(self.qnet.logit_scale.exp().clamp(max=100.))
        b=torch.load(bundle,map_location='cpu',weights_only=False)
        self.head_kind=b['kind'];self.threshold=float(b['threshold'])
        self.grip_head=H.KC1Head(int(b['input_dim'])).to(device);self.grip_head.load_state_dict(b['state_dict']);self.grip_head.eval()
        self.query_vlm=self.query=='iii'
        self.cal=json.loads((K.OUT/'calibration.json').read_text())
        self.b=float(self.cal['b'])
        if self.query_vlm!=(p.vlm is not None):raise AssertionError('KC1 selected query and VLM path mismatch')

    def infer(self,observation):
        p=self.impl;torch,jax,jnp=p.torch,p.jax,p.jnp;started=time.monotonic()
        row,obs=p._obs(observation);rng_key=jax.random.key(int(observation['s2_rng_seed']))
        (x,valid),sn,an,step_t=p.tokens(observation)
        vlm=None
        if self.query_vlm:
            f={k:np.asarray(v) for k,v in p.vlm(obs).items()}
            vlm=np.zeros((1,34,2048),np.float16);vlm[0,:32]=f['vlm_img'][0]
            vlm[0,32]=f['vlm_txt_mean'][0].astype(np.float16);vlm[0,33]=f['vlm_txt_last'][0].astype(np.float16)
        T=lambda a,dt=torch.float32:torch.as_tensor(a,device=p.device,dtype=dt)
        split=C.InlineSplit(T(x[None]),T(valid[None],torch.bool),None if vlm is None else T(vlm,torch.float16))
        with torch.no_grad():q=self.qnet(split,T([0],torch.long))
        fake={'task':np.asarray([-1]),'episode':np.asarray([-1])}
        feat=H.retrieval_features(self.qnet,split,fake,self.keys,np.asarray(p.unit_task),np.full(len(p.unit_task),-2))
        ids=np.asarray(feat['top_ids'][0],np.int64);logits=np.asarray(feat['top_logits'][0],np.float64)
        top10=ids[:10]
        disagreement,weights=topk_disagreement(np.asarray(p.units[top10][None],np.float32),logits[None,:10],k=10)
        d=float(disagreement[0]);t0=float(t0_from_disagreement(d,self.cal));fallback=bool(d>float(self.cal['D_b']))
        last_valid=len(an)>0;last_cmd=float(an[-1,6]) if last_valid else C.DEFAULT_GRIP
        last_sign=1. if last_valid and last_cmd>0 else -1.
        gx=C.grip_extra(np.asarray([last_cmd]),np.asarray([last_valid]),sn[-1:,6:8])
        xin=H.head_input(feat,gx,self.head_kind)
        with torch.no_grad():pclose=torch.sigmoid(self.grip_head(torch.as_tensor(xin,device=p.device))).cpu().numpy()
        decoded=C.decode_gripper(pclose,np.asarray([last_sign]),self.threshold)[0]
        unit=int(top10[0]);draft=np.zeros((1,10,32),np.float32);draft[0,:,:6]=p.units[unit,:,:6];draft[0,:,6]=decoded
        if fallback:
            output=np.asarray(p.normal(rng_key,obs,num_steps=10),np.float32).copy()
        else:
            noise=jax.random.normal(jax.random.fold_in(rng_key,7),(1,10,32))
            output=np.asarray(p.bridge(jax.random.key(0),obs,jnp.asarray(draft),jnp.asarray([t0]),num_steps=10,noise=noise),np.float32).copy()
        physical=p.unnormalize({'actions':output[0],'state':row['state']})['actions'][:,:7]
        if not np.isfinite(physical).all():raise FloatingPointError('nonfinite KC1 action')
        return {'actions':physical.astype(np.float32),'s2_debug':{'runtime':'kc1','query':self.query,'unit_id':unit,
            'top10_unit_ids':top10.tolist(),'top10_probabilities':weights[0].tolist(),'disagreement_D':d,
            'D_b':float(self.cal['D_b']),'t0':t0,'fallback_pi05':fallback,'gripper_head':self.head_kind,
            'gripper_chunk':output[0,:,6].astype(float).tolist(),'decoder_pclose':pclose[0].astype(float).tolist(),
            'step_t':step_t,'history_valid_steps':int(valid.sum()),'seconds':time.monotonic()-started}}


def main():
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--mode',choices=('memory',),required=True)
    ap.add_argument('--query',choices=('iii','i'),required=True);ap.add_argument('--expert-checkpoint',required=True)
    ap.add_argument('--bundle',required=True);ap.add_argument('--tpol',default='{}');ap.add_argument('--port',type=int,required=True)
    a=ap.parse_args();policy=KC1Policy(mode=a.mode,query=a.query,expert_checkpoint=a.expert_checkpoint,bundle=a.bundle,tpol=json.loads(a.tpol))
    WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata={'experiment':'KC1','query':policy.query,'horizon':10}).serve_forever()

if __name__=='__main__':main()
