"""M1: continuous 12-D base + 4-D residual with covariance and HSIC penalties."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from ._common import seed,stats,norm,calibrate,memory_add,memory_query,tensor,batch_iter,validation_split,update_early_stopping,AdditiveActionDecoder


def _hsic(x,y):
    # Biased RBF-HSIC, O(batch^2); bounded training minibatches keep this cheap.
    if x.shape[0] < 2: return x.sum()*0
    def kernel(a):
        d2=torch.cdist(a,a).square(); med=torch.median(d2.detach().flatten()).clamp_min(1e-6)
        return torch.exp(-d2/med)
    n=x.shape[0]; h=torch.eye(n,device=x.device)-torch.ones((n,n),device=x.device)/n
    k=kernel(x); l=kernel(y)
    return (h@k@h*l).sum()/((n-1)**2)


class M1:
    def __init__(self,*,independence_weight=.01,beta_r=1.0,base_use_weight=5.,base_use_margin=.05,width=128,epochs=10,batch_size=128,device="cpu",seed_value=0,decoder_obs=False,base_dim=12,residual_dim=4):
        if int(base_dim)<1 or int(residual_dim)<0: raise ValueError("base_dim must be positive and residual_dim non-negative")
        self.independence_weight=independence_weight; self.beta_r=float(beta_r); self.decoder_obs=bool(decoder_obs); self.base_use_weight=float(base_use_weight); self.base_use_margin=float(base_use_margin); self.width=width; self.epochs=epochs; self.batch_size=batch_size; self.device=torch.device(device); self.seed_value=seed_value
        self.memory=[]; self.topk=5; self.temperature=.5; self.base_dim=int(base_dim); self.residual_dim=int(residual_dim)

    def fit(self,train):
        seed(int(train.get("seed",self.seed_value)),str(self.device))
        a=np.asarray(train["actions"],np.float32); s=np.asarray(train["state"],np.float32); f=np.asarray(train["feature"],np.float32)
        self.action_shape=a.shape[1:]; self.a_dim=int(np.prod(self.action_shape)); self.state_dim=s.shape[1]; self.feature_dim=f.shape[1]
        self.am,self.ass=stats(a.reshape(len(a),-1)); self.sm,self.ss=stats(s); self.fm,self.fs=stats(f)
        an=norm(a.reshape(len(a),-1),self.am,self.ass); sn=norm(s,self.sm,self.ss); fn=norm(f,self.fm,self.fs)
        self.encoder=nn.Sequential(nn.Linear(self.a_dim+self.state_dim,self.width),nn.GELU(),nn.Linear(self.width,self.width),nn.GELU(),nn.Linear(self.width,self.base_dim+2*self.residual_dim)).to(self.device)
        self.prior_net=nn.Sequential(nn.Linear(self.feature_dim+self.state_dim,self.width),nn.GELU(),nn.Linear(self.width,self.width),nn.GELU(),nn.Linear(self.width,self.base_dim+self.residual_dim)).to(self.device)
        context_dim=self.state_dim+(self.feature_dim if self.decoder_obs else 0)
        self.base_decoder=nn.Sequential(nn.Linear(self.base_dim+context_dim,self.width),nn.GELU(),nn.Linear(self.width,self.width),nn.GELU(),nn.Linear(self.width,self.a_dim)).to(self.device)
        self.residual_decoder=nn.Sequential(nn.Linear(self.residual_dim,self.width),nn.GELU(),nn.Linear(self.width,self.width),nn.GELU(),nn.Linear(self.width,self.a_dim)).to(self.device)
        self.decoder=AdditiveActionDecoder(self.base_decoder,self.residual_decoder,base_dim=self.base_dim,residual_dim=self.residual_dim,state_dim=self.state_dim,feature_dim=self.feature_dim,decoder_obs=self.decoder_obs,residual_uses_context=False).to(self.device)
        opt=torch.optim.AdamW(list(self.encoder.parameters())+list(self.prior_net.parameters())+list(self.base_decoder.parameters())+list(self.residual_decoder.parameters()),lr=1e-3,weight_decay=1e-5)
        at=tensor(an,self.device); st=tensor(sn,self.device); ft=tensor(fn,self.device); rng=np.random.default_rng(train.get("seed",self.seed_value))
        train_ix,val_ix=validation_split(len(a),train.get("seed",self.seed_value))
        val=torch.as_tensor(val_ix,device=self.device); best=float("inf"); stale=0; patience=int(train.get("early_stopping_patience",3)); self.epochs_run=0
        for epoch in range(self.epochs):
            for local_ix in batch_iter(len(train_ix),self.batch_size,rng):
                ix=train_ix[local_ix]
                ii=torch.as_tensor(ix,device=self.device); latent=self.encoder(torch.cat([at[ii],st[ii]],1)); b=latent[:,:self.base_dim]; r_mu,r_logvar=self._residual_stats(latent[:,self.base_dim:]); r=r_mu+torch.randn_like(r_mu)*torch.exp(.5*r_logvar)
                pred_base,pred_res=self.prior_net(torch.cat([ft[ii],st[ii]],1)).split((self.base_dim,self.residual_dim),dim=1)
                base_pred=self._decode_base(b,ft[ii],st[ii]); residual_pred=self._decode_residual(r,ft[ii],st[ii],b); pred=base_pred+residual_pred
                recon=(pred-at[ii]).square().mean()
                zero_pred=self._decode_base(torch.zeros_like(b),ft[ii],st[ii])+self._decode_residual(r,ft[ii],st[ii],torch.zeros_like(b))
                base_only=base_pred
                base_use=torch.relu(self.base_use_margin+recon-(zero_pred-at[ii]).square().mean())
                prior_loss=(pred_base-b.detach()).square().mean()+(pred_res-r_mu.detach()).square().mean()
                kl_r=.5*(r_mu.square()+r_logvar.exp()-r_logvar-1).sum(1).mean()
                bc=b-b.mean(0); rc=r_mu-r_mu.mean(0); cov=(bc.T@rc/max(len(ix)-1,1)).square().mean()
                hsic=_hsic(b,r_mu)
                loss=recon+.25*prior_loss+self.independence_weight*(cov+hsic)+self.beta_r*kl_r+self.base_use_weight*(base_use+.5*(base_only-at[ii]).square().mean())
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(self.parameters(),1.0); opt.step()
            self.epochs_run=epoch+1
            if len(val):
                with torch.no_grad():
                    latent=self.encoder(torch.cat([at[val],st[val]],1)); vb=latent[:,:self.base_dim]; vr,_=self._residual_stats(latent[:,self.base_dim:])
                    vp=self._decode_base(vb,ft[val],st[val])+self._decode_residual(vr,ft[val],st[val],vb)
                    val_loss=(vp-at[val]).square().mean().item()
                best,stale,stop=update_early_stopping(val_loss,best,stale,patience=patience)
                if stop: break
        self.encoder.eval(); self.prior_net.eval(); self.base_decoder.eval(); self.residual_decoder.eval()
        self.tau,self.d_ref=calibrate(train["key"],train["episode_id"],train["init_state_id"],quantile=.95)

    def parameters(self): return list(self.encoder.parameters())+list(self.prior_net.parameters())+list(self.base_decoder.parameters())+list(self.residual_decoder.parameters())

    def encode(self,A,s):
        an=norm(np.asarray(A).reshape(len(A),-1),self.am,self.ass); sn=norm(s,self.sm,self.ss)
        with torch.no_grad(): z=self.encoder(torch.cat([tensor(an,self.device),tensor(sn,self.device)],1)); r,_=self._residual_stats(z[:,self.base_dim:])
        return z[:,:self.base_dim].cpu().numpy(),r.cpu().numpy()

    def prior(self,f,s):
        fn=norm(f,self.fm,self.fs); sn=norm(s,self.sm,self.ss)
        with torch.no_grad(): z=self.prior_net(torch.cat([tensor(fn,self.device),tensor(sn,self.device)],1))
        return z[:,:self.base_dim].cpu().numpy(),z[:,self.base_dim:].cpu().numpy()

    def _residual_stats(self,x):
        mu,logvar=x.chunk(2,dim=-1); return mu,logvar.clamp(-8,5)

    def residual_kl_bits(self,A,s):
        an=norm(np.asarray(A).reshape(len(A),-1),self.am,self.ass); sn=norm(s,self.sm,self.ss)
        with torch.no_grad():
            z=self.encoder(torch.cat([tensor(an,self.device),tensor(sn,self.device)],1)); mu,lv=self._residual_stats(z[:,self.base_dim:]); kl=.5*(mu.square()+lv.exp()-lv-1).sum(1)
        return (kl/np.log(2)).cpu().numpy()

    def _context(self,f,s): return torch.cat([*(([f] if self.decoder_obs else [])),s],1)
    def _decode_base(self,b,f,s): return self.base_decoder(torch.cat([b,self._context(f,s)],1))
    def _decode_residual(self,r,f,s,b):
        return self.residual_decoder(r)

    def decode(self,b,r,f,s):
        fn=norm(f,self.fm,self.fs); sn=norm(s,self.sm,self.ss)
        with torch.no_grad(): bt=tensor(b,self.device); y=(self._decode_base(bt,tensor(fn,self.device),tensor(sn,self.device))+self._decode_residual(tensor(r,self.device),tensor(fn,self.device),tensor(sn,self.device),bt)).cpu().numpy()
        return (y*self.ass+self.am).reshape(len(y),*self.action_shape)

    def memory_add(self,k,b,meta): return memory_add(self.memory,k,b,meta,self.tau)
    def memory_query(self,k,meta=None): return memory_query(self.memory,k,meta,self.topk,self.temperature,self.d_ref)
    def blend_base(self,candidates,weights): return np.asarray(weights)@np.asarray(candidates)
    def base_features(self,b): return np.asarray(b,dtype=np.float32)
    def describe(self): return {"name":"M1","base":f"continuous {self.base_dim}-D","residual":f"Gaussian {self.residual_dim}-D bottleneck","residual_decoder_input":"residual only; proprioception and observations cannot bypass the residual bottleneck","objective":"action MSE + 0.25 prior regression + lambda*(cross-covariance penalty + RBF-HSIC) + beta_r*KL(q(r|A,s)||N(0,I)) + base-use margin penalty","independence_weight":self.independence_weight,"beta_r":self.beta_r,"base_dim":self.base_dim,"residual_dim":self.residual_dim,"base_use_weight":self.base_use_weight,"base_use_margin":self.base_use_margin,"decoder_obs":self.decoder_obs,"decoder_observation_conditioning":"comparison only; default is false","weights_compared":[0.0,0.01,0.1],"residual_kl_weights_compared":[1.0,10.0],"weight_reason":"strong standard-normal KL weights constrain residual capacity so base remains behaviorally necessary","updates_without_gradient":"none during memory insertion","neural_weights_fixed_on_memory_add":True,"memory_representatives_change_on_add":False,"old_ids_change_on_add":False}


def build_variants():
    variants={}
    for w in (0.0,0.01,0.1):
        variants[f"M1_hsic{w:g}"]=M1(independence_weight=w,beta_r=1.0)
        variants[f"M1_hsic{w:g}_br10"]=M1(independence_weight=w,beta_r=10.0)
    variants["M1_hsic0.01_decoder_obs"]=M1(independence_weight=.01,beta_r=1.0,decoder_obs=True)
    return variants
