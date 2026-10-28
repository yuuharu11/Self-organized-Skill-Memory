"""M0: current 32-D continuous Gaussian skill; no residual branch."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from ._common import seed,stats,norm,calibrate,memory_add,memory_query,tensor,batch_iter,validation_split,update_early_stopping


class M0:
    def __init__(self,*,beta=.01,base_use_weight=5.,base_use_margin=.05,width=128,epochs=10,batch_size=128,device="cpu",seed_value=0,decoder_obs=False):
        self.beta=beta; self.width=width; self.epochs=epochs; self.batch_size=batch_size; self.device=torch.device(device); self.seed_value=seed_value
        self.decoder_obs=bool(decoder_obs); self.base_use_weight=float(base_use_weight); self.base_use_margin=float(base_use_margin)
        self.memory=[]; self.topk=5; self.temperature=.5

    def fit(self,train):
        seed(int(train.get("seed",self.seed_value)),str(self.device))
        a=np.asarray(train["actions"],np.float32); s=np.asarray(train["state"],np.float32); f=np.asarray(train["feature"],np.float32)
        self.action_shape=a.shape[1:]; self.a_dim=int(np.prod(self.action_shape)); self.state_dim=s.shape[1]; self.feature_dim=f.shape[1]
        self.am,self.ass=stats(a.reshape(len(a),-1)); self.sm,self.ss=stats(s); self.fm,self.fs=stats(f)
        an=norm(a.reshape(len(a),-1),self.am,self.ass); sn=norm(s,self.sm,self.ss); fn=norm(f,self.fm,self.fs)
        self.encoder=nn.Sequential(nn.Linear(self.a_dim+self.state_dim,self.width),nn.GELU(),nn.Linear(self.width,self.width),nn.GELU(),nn.Linear(self.width,64)).to(self.device)
        self.prior_net=nn.Sequential(nn.Linear(self.feature_dim+self.state_dim,self.width),nn.GELU(),nn.Linear(self.width,self.width),nn.GELU(),nn.Linear(self.width,64)).to(self.device)
        self.decoder=nn.Sequential(nn.Linear(32+self.state_dim+(self.feature_dim if self.decoder_obs else 0),self.width),nn.GELU(),nn.Linear(self.width,self.width),nn.GELU(),nn.Linear(self.width,self.a_dim)).to(self.device)
        opt=torch.optim.AdamW(list(self.encoder.parameters())+list(self.prior_net.parameters())+list(self.decoder.parameters()),lr=1e-3,weight_decay=1e-5)
        at=tensor(an,self.device); st=tensor(sn,self.device); ft=tensor(fn,self.device); rng=np.random.default_rng(train.get("seed",self.seed_value))
        train_ix,val_ix=validation_split(len(a),train.get("seed",self.seed_value))
        val=torch.as_tensor(val_ix,device=self.device); best=float("inf"); stale=0; patience=int(train.get("early_stopping_patience",3)); self.epochs_run=0
        for epoch in range(self.epochs):
            for local_ix in batch_iter(len(train_ix),self.batch_size,rng):
                ix=train_ix[local_ix]
                ix_t=torch.as_tensor(ix,device=self.device); mu,lv=self._split(self.encoder(torch.cat([at[ix_t],st[ix_t]],1)))
                pm,plv=self._split(self.prior_net(torch.cat([ft[ix_t],st[ix_t]],1)))
                z=mu+torch.randn_like(mu)*torch.exp(.5*lv)
                pred=self.decoder(self._decoder_input(z,ft[ix_t],st[ix_t]))
                recon=(pred-at[ix_t]).square().mean()
                zero_pred=self.decoder(self._decoder_input(torch.zeros_like(z),ft[ix_t],st[ix_t]))
                base_use= torch.relu(self.base_use_margin+recon-(zero_pred-at[ix_t]).square().mean())
                kl=.5*(plv-lv+(lv.exp()+(mu-pm).square())/plv.exp()-1).sum(1).mean()
                loss=recon+self.beta*kl+self.base_use_weight*(base_use+(zero_pred-at[ix_t]).square().mean())
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(self.parameters(),1.0); opt.step()
            self.epochs_run=epoch+1
            if len(val):
                with torch.no_grad():
                    vm,_=self._split(self.encoder(torch.cat([at[val],st[val]],1)))
                    vp=self.decoder(self._decoder_input(vm,ft[val],st[val]))
                    val_loss=(vp-at[val]).square().mean().item()
                best,stale,stop=update_early_stopping(val_loss,best,stale,patience=patience)
                if stop: break
        self.encoder.eval(); self.prior_net.eval(); self.decoder.eval()
        self.tau,self.d_ref=calibrate(train["key"],train["episode_id"],train["init_state_id"],quantile=.95)

    @staticmethod
    def _split(x):
        a,b=x.chunk(2,dim=-1); return a,b.clamp(-8,5)

    def parameters(self): return list(self.encoder.parameters())+list(self.prior_net.parameters())+list(self.decoder.parameters())

    def _decoder_input(self,b,f,s): return torch.cat([b,*(([f] if self.decoder_obs else [])),s],1)

    def encode(self,A,s):
        a=norm(np.asarray(A).reshape(len(A),-1),self.am,self.ass); sn=norm(s,self.sm,self.ss)
        with torch.no_grad(): b,_=self._split(self.encoder(torch.cat([tensor(a,self.device),tensor(sn,self.device)],1)))
        return b.cpu().numpy(),None

    def prior(self,f,s):
        fn=norm(f,self.fm,self.fs); sn=norm(s,self.sm,self.ss)
        with torch.no_grad(): b,_=self._split(self.prior_net(torch.cat([tensor(fn,self.device),tensor(sn,self.device)],1)))
        return b.cpu().numpy(),None

    def kl_rate_bits(self,A,s,f):
        """Per-window q(z|A,s)||p(z|f,s) KL, expressed in bits."""
        an=norm(np.asarray(A).reshape(len(A),-1),self.am,self.ass); sn=norm(s,self.sm,self.ss); fn=norm(f,self.fm,self.fs)
        with torch.no_grad():
            qm,ql=self._split(self.encoder(torch.cat([tensor(an,self.device),tensor(sn,self.device)],1)))
            pm,pl=self._split(self.prior_net(torch.cat([tensor(fn,self.device),tensor(sn,self.device)],1)))
            kl=.5*(pl-ql+(ql.exp()+(qm-pm).square())/pl.exp()-1).sum(1)
        return (kl / math.log(2)).cpu().numpy()

    def decode(self,b,r,f,s):
        b=np.asarray(b,dtype=np.float32); fn=norm(f,self.fm,self.fs); sn=norm(s,self.sm,self.ss)
        with torch.no_grad(): y=self.decoder(self._decoder_input(tensor(b,self.device),tensor(fn,self.device),tensor(sn,self.device))).cpu().numpy()
        return (y*self.ass+self.am).reshape(len(y),*self.action_shape)

    def memory_add(self,k,b,meta): return memory_add(self.memory,k,b,meta,self.tau)
    def memory_query(self,k,meta=None): return memory_query(self.memory,k,meta,self.topk,self.temperature,self.d_ref)
    def blend_base(self,candidates,weights): return np.asarray(weights)@np.asarray(candidates)
    def base_features(self,b): return np.asarray(b,dtype=np.float32)
    def describe(self): return {"name":"M0","base":"32-D Gaussian posterior mean z","residual":"none","objective":"normalized action MSE + beta*KL(q||p) + base-use margin penalty","beta":self.beta,"base_use_weight":self.base_use_weight,"base_use_margin":self.base_use_margin,"decoder_obs":self.decoder_obs,"decoder_observation_conditioning":"comparison only; default is false","beta_reason":"0.01 matches the frozen light-model reference","updates_without_gradient":"none during memory insertion","neural_weights_fixed_on_memory_add":True,"memory_representatives_change_on_add":False,"old_ids_change_on_add":False}


def build_variants():
    return {"M0":M0(),"M0_decoder_obs":M0(decoder_obs=True)}
