"""M2: EMA vector-quantized base (16/32/64 codes) + 4-D continuous residual."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ._common import seed,stats,norm,calibrate,memory_add,memory_query,tensor,batch_iter,validation_split,update_early_stopping,AdditiveActionDecoder


class M2:
    def __init__(self,*,n_codes=32,embedding_dim=16,residual_dim=4,beta_r=1.0,decoder_obs=False,base_use_weight=5.,base_use_margin=.05,width=128,epochs=10,batch_size=128,device="cpu",seed_value=0,ema_decay=.99,dead_threshold=.1):
        if n_codes not in (16,32,64): raise ValueError("n_codes must be 16, 32, or 64")
        self.n_codes=n_codes; self.embedding_dim=embedding_dim; self.residual_dim=residual_dim; self.beta_r=float(beta_r); self.decoder_obs=bool(decoder_obs); self.base_use_weight=float(base_use_weight); self.base_use_margin=float(base_use_margin); self.width=width; self.epochs=epochs; self.batch_size=batch_size; self.device=torch.device(device); self.seed_value=seed_value; self.ema_decay=ema_decay; self.dead_threshold=dead_threshold
        self.memory=[]; self.topk=5; self.temperature=.5

    def fit(self,train):
        seed(int(train.get("seed",self.seed_value)),str(self.device))
        a=np.asarray(train["actions"],np.float32); s=np.asarray(train["state"],np.float32); f=np.asarray(train["feature"],np.float32)
        self.action_shape=a.shape[1:]; self.a_dim=int(np.prod(self.action_shape)); self.state_dim=s.shape[1]; self.feature_dim=f.shape[1]
        self.am,self.ass=stats(a.reshape(len(a),-1)); self.sm,self.ss=stats(s); self.fm,self.fs=stats(f)
        an=norm(a.reshape(len(a),-1),self.am,self.ass); sn=norm(s,self.sm,self.ss); fn=norm(f,self.fm,self.fs)
        self.encoder=nn.Sequential(nn.Linear(self.a_dim+self.state_dim,width:=self.width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,self.embedding_dim+2*self.residual_dim)).to(self.device)
        self.prior_net=nn.Sequential(nn.Linear(self.feature_dim+self.state_dim,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,self.n_codes+self.residual_dim)).to(self.device)
        context_dim=self.state_dim+(self.feature_dim if self.decoder_obs else 0)
        self.base_decoder=nn.Sequential(nn.Linear(self.embedding_dim+context_dim,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,self.a_dim)).to(self.device)
        self.residual_decoder=nn.Sequential(nn.Linear(self.residual_dim,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,self.a_dim)).to(self.device)
        self.decoder=AdditiveActionDecoder(self.base_decoder,self.residual_decoder,base_dim=self.embedding_dim,residual_dim=self.residual_dim,state_dim=self.state_dim,feature_dim=self.feature_dim,decoder_obs=self.decoder_obs,residual_uses_context=False).to(self.device)
        self.codebook=torch.randn(self.n_codes,self.embedding_dim,device=self.device)*.05
        self.ema_count=torch.ones(self.n_codes,device=self.device); self.ema_sum=self.codebook.clone()
        opt=torch.optim.AdamW(list(self.encoder.parameters())+list(self.prior_net.parameters())+list(self.base_decoder.parameters())+list(self.residual_decoder.parameters()),lr=1e-3,weight_decay=1e-5)
        at=tensor(an,self.device); st=tensor(sn,self.device); ft=tensor(fn,self.device); rng=np.random.default_rng(train.get("seed",self.seed_value))
        train_ix,val_ix=validation_split(len(a),train.get("seed",self.seed_value))
        val=torch.as_tensor(val_ix,device=self.device); best=float("inf"); stale=0; patience=int(train.get("early_stopping_patience",3)); self.epochs_run=0
        for epoch in range(self.epochs):
            for local_ix in batch_iter(len(train_ix),self.batch_size,rng):
                ix=train_ix[local_ix]
                ii=torch.as_tensor(ix,device=self.device); enc=self.encoder(torch.cat([at[ii],st[ii]],1)); ze=enc[:,:self.embedding_dim]; r_mu,r_logvar=self._residual_stats(enc[:,self.embedding_dim:]); r=r_mu+torch.randn_like(r_mu)*torch.exp(.5*r_logvar)
                d2=torch.cdist(ze,self.codebook).square(); codes=d2.argmin(1); q=self.codebook[codes]
                # EMA codebook updates are explicitly non-gradient updates.
                with torch.no_grad():
                    counts=F.one_hot(codes,self.n_codes).float().sum(0); sums=F.one_hot(codes,self.n_codes).float().T@ze.detach()
                    self.ema_count.mul_(self.ema_decay).add_(counts,alpha=1-self.ema_decay)
                    self.ema_sum.mul_(self.ema_decay).add_(sums,alpha=1-self.ema_decay)
                    self.codebook.copy_(self.ema_sum/(self.ema_count[:,None]+1e-5))
                    dead=self.ema_count<self.dead_threshold
                    if dead.any():
                        repl=ze.detach()[torch.randint(len(ze),(int(dead.sum()),),device=self.device)]
                        self.codebook[dead]=repl; self.ema_sum[dead]=repl; self.ema_count[dead]=1.0
                q_st=ze+(q-ze).detach(); logits_and_r=self.prior_net(torch.cat([ft[ii],st[ii]],1)); logits=logits_and_r[:,:self.n_codes]; rp=logits_and_r[:,self.n_codes:]
                base_pred=self._decode_base(q_st,ft[ii],st[ii]); residual_pred=self._decode_residual(r,ft[ii],st[ii],q_st); pred=base_pred+residual_pred
                recon=(pred-at[ii]).square().mean()
                pred_zero=self._decode_base(torch.zeros_like(q_st),ft[ii],st[ii])+self._decode_residual(r,ft[ii],st[ii],torch.zeros_like(q_st))
                base_only=base_pred
                base_use=torch.relu(self.base_use_margin+recon-(pred_zero-at[ii]).square().mean())
                commitment=(ze-q.detach()).square().mean(); prior_loss=F.cross_entropy(logits,codes)+.25*(rp-r_mu.detach()).square().mean()
                kl_r=.5*(r_mu.square()+r_logvar.exp()-r_logvar-1).sum(1).mean()
                loss=recon+.25*commitment+.25*prior_loss+self.beta_r*kl_r+self.base_use_weight*(base_use+.5*(base_only-at[ii]).square().mean())
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(self.parameters(),1.0); opt.step()
            self.epochs_run=epoch+1
            if len(val):
                with torch.no_grad():
                    enc=self.encoder(torch.cat([at[val],st[val]],1)); ze=enc[:,:self.embedding_dim]; rv,_=self._residual_stats(enc[:,self.embedding_dim:])
                    codes=torch.cdist(ze,self.codebook).argmin(1); qv=self.codebook[codes]
                    vp=self._decode_base(qv,ft[val],st[val])+self._decode_residual(rv,ft[val],st[val],qv)
                    val_loss=(vp-at[val]).square().mean().item()
                best,stale,stop=update_early_stopping(val_loss,best,stale,patience=patience)
                if stop: break
        self.encoder.eval(); self.prior_net.eval(); self.base_decoder.eval(); self.residual_decoder.eval()
        self.tau,self.d_ref=calibrate(train["key"],train["episode_id"],train["init_state_id"],quantile=.95)

    def parameters(self): return list(self.encoder.parameters())+list(self.prior_net.parameters())+list(self.base_decoder.parameters())+list(self.residual_decoder.parameters())

    def encode(self,A,s):
        an=norm(np.asarray(A).reshape(len(A),-1),self.am,self.ass); sn=norm(s,self.sm,self.ss)
        with torch.no_grad(): enc=self.encoder(torch.cat([tensor(an,self.device),tensor(sn,self.device)],1)); ze=enc[:,:self.embedding_dim]; r,_=self._residual_stats(enc[:,self.embedding_dim:]); codes=torch.cdist(ze,self.codebook).argmin(1)
        return codes.cpu().numpy().astype(np.int64),r.cpu().numpy()

    def prior(self,f,s):
        fn=(norm(f,self.fm,self.fs) if self.feature_dim else np.zeros((len(s),0),np.float32)); sn=norm(s,self.sm,self.ss)
        with torch.no_grad(): out=self.prior_net(torch.cat([tensor(fn,self.device),tensor(sn,self.device)],1))
        return out[:,:self.n_codes].cpu().numpy(),out[:,self.n_codes:].cpu().numpy()

    def prior_base(self,base_distribution): return np.asarray(base_distribution).argmax(axis=-1).astype(np.int64)

    def _residual_stats(self,x):
        mu,logvar=x.chunk(2,dim=-1); return mu,logvar.clamp(-8,5)

    def residual_kl_bits(self,A,s):
        an=norm(np.asarray(A).reshape(len(A),-1),self.am,self.ass); sn=norm(s,self.sm,self.ss)
        with torch.no_grad():
            enc=self.encoder(torch.cat([tensor(an,self.device),tensor(sn,self.device)],1)); mu,lv=self._residual_stats(enc[:,self.embedding_dim:]); kl=.5*(mu.square()+lv.exp()-lv-1).sum(1)
        return (kl/np.log(2)).cpu().numpy()

    def base_features(self,b):
        ids=torch.as_tensor(np.asarray(b,dtype=np.int64).reshape(-1),device=self.device)
        with torch.no_grad(): return self.codebook[ids].cpu().numpy()

    def decode(self,b,r,f,s):
        bb=np.asarray(b)
        if bb.ndim==1 and np.issubdtype(bb.dtype,np.integer): bb=self.base_features(bb)
        fn=norm(f,self.fm,self.fs); sn=norm(s,self.sm,self.ss)
        with torch.no_grad(): bt=tensor(bb,self.device); y=(self._decode_base(bt,tensor(fn,self.device),tensor(sn,self.device))+self._decode_residual(tensor(r,self.device),tensor(fn,self.device),tensor(sn,self.device),bt)).cpu().numpy()
        return (y*self.ass+self.am).reshape(len(y),*self.action_shape)

    def _context(self,f,s): return torch.cat([*(([f] if self.decoder_obs else [])),s],1)
    def _decode_base(self,b,f,s): return self.base_decoder(torch.cat([F.normalize(b,dim=1,eps=1e-6),self._context(f,s)],1))
    def _decode_residual(self,r,f,s,b):
        return self.residual_decoder(r)

    def memory_add(self,k,b,meta): return memory_add(self.memory,k,b,meta,self.tau)
    def memory_query(self,k,meta=None): return memory_query(self.memory,k,meta,self.topk,self.temperature,self.d_ref)
    def blend_base(self,candidates,weights): return np.asarray(weights)@self.base_features(candidates)
    def describe(self): return {"name":"M2","base":f"EMA VQ code with {self.n_codes} entries, {self.embedding_dim}-D embedding","residual":f"Gaussian {self.residual_dim}-D bottleneck","residual_decoder_input":"residual only; proprioception and observations cannot bypass the residual bottleneck","objective":"normalized action MSE + commitment + prior CE/residual regression + beta_r*KL(q(r|A,s)||N(0,I)) + base-use margin penalty","beta_r":self.beta_r,"base_use_weight":self.base_use_weight,"base_use_margin":self.base_use_margin,"decoder_obs":self.decoder_obs,"decoder_observation_conditioning":"comparison only; default is false","residual_kl_weights_compared":[1.0,10.0],"ema_decay":self.ema_decay,"unused_code_reinitialization":"codes with EMA count < 0.1 are reset to random current-batch encoder vectors","updates_without_gradient":"codebook EMA and dead-code reinitialization during fit; no updates when spatial memory rows are added","neural_weights_fixed_on_memory_add":True,"memory_representatives_change_on_add":False,"old_ids_change_on_add":False}


def build_variants():
    variants={}
    for k in (16,32,64):
        variants[f"M2_vq{k}"]=M2(n_codes=k,beta_r=1.0)
        variants[f"M2_vq{k}_br10"]=M2(n_codes=k,beta_r=10.0)
    variants["M2_vq32_decoder_obs"]=M2(n_codes=32,beta_r=1.0,decoder_obs=True)
    return variants
