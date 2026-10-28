"""Finite Scalar Quantization base with a continuous 4-D residual.

This is an FSQ-style small-codebook comparator, not a reproduction of the
complete QueST system. Each bit is a scalar with fixed levels {-1,+1}; the
Cartesian product gives exactly 16, 32, or 64 base codes.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ._common import seed, stats, norm, calibrate, memory_add, memory_query, tensor, batch_iter, validation_split, update_early_stopping, AdditiveActionDecoder


class FSQ:
    def __init__(self, *, n_codes=32, residual_dim=4, beta_r=1.0, decoder_obs=False, base_use_weight=5., base_use_margin=.05, width=128, epochs=10,
                 batch_size=128, device="cpu", seed_value=0):
        if n_codes not in (16, 32, 64):
            raise ValueError("FSQ code count must be 16, 32, or 64")
        self.n_codes = n_codes
        self.base_dim = int(math.log2(n_codes))
        self.residual_dim = residual_dim
        self.beta_r, self.decoder_obs = float(beta_r), bool(decoder_obs)
        self.base_use_weight,self.base_use_margin=float(base_use_weight),float(base_use_margin)
        self.width, self.epochs, self.batch_size = width, epochs, batch_size
        self.device, self.seed_value = torch.device(device), seed_value
        self.memory = []
        self.topk, self.temperature = 5, .5

    def _quantize(self, x):
        hard = torch.where(torch.tanh(x) >= 0, 1.0, -1.0)
        return x + (hard - x).detach()

    def _to_ids(self, q):
        bits = (q > 0).to(torch.long)
        powers = (2 ** torch.arange(self.base_dim, device=q.device)).view(1, -1)
        return (bits * powers).sum(-1)

    def _from_ids(self, ids):
        ids = torch.as_tensor(np.asarray(ids, dtype=np.int64).reshape(-1), device=self.device)
        powers = 2 ** torch.arange(self.base_dim, device=self.device)
        return (((ids[:, None] >> powers[None, :]) & 1).float() * 2.0 - 1.0)

    def fit(self, train):
        seed(int(train.get("seed", self.seed_value)), str(self.device))
        a = np.asarray(train["actions"], np.float32)
        s = np.asarray(train["state"], np.float32)
        f = np.asarray(train["feature"], np.float32)
        self.action_shape = a.shape[1:]
        self.a_dim = int(np.prod(self.action_shape))
        self.state_dim, self.feature_dim = s.shape[1], f.shape[1]
        self.am, self.ass = stats(a.reshape(len(a), -1))
        self.sm, self.ss = stats(s)
        self.fm, self.fs = stats(f)
        an = norm(a.reshape(len(a), -1), self.am, self.ass)
        sn, fn = norm(s, self.sm, self.ss), norm(f, self.fm, self.fs)
        self.encoder = nn.Sequential(nn.Linear(self.a_dim + self.state_dim, self.width), nn.GELU(),
                                     nn.Linear(self.width, self.width), nn.GELU(),
                                     nn.Linear(self.width, self.base_dim + 2*self.residual_dim)).to(self.device)
        self.prior_net = nn.Sequential(nn.Linear(self.feature_dim + self.state_dim, self.width), nn.GELU(),
                                       nn.Linear(self.width, self.width), nn.GELU(),
                                       nn.Linear(self.width, self.n_codes + self.residual_dim)).to(self.device)
        context_dim=self.state_dim+(self.feature_dim if self.decoder_obs else 0)
        self.base_decoder=nn.Sequential(nn.Linear(self.base_dim+context_dim,self.width),nn.GELU(),nn.Linear(self.width,self.width),nn.GELU(),nn.Linear(self.width,self.a_dim)).to(self.device)
        self.residual_decoder=nn.Sequential(nn.Linear(self.residual_dim,self.width),nn.GELU(),nn.Linear(self.width,self.width),nn.GELU(),nn.Linear(self.width,self.a_dim)).to(self.device)
        self.decoder=AdditiveActionDecoder(self.base_decoder,self.residual_decoder,base_dim=self.base_dim,residual_dim=self.residual_dim,state_dim=self.state_dim,feature_dim=self.feature_dim,decoder_obs=self.decoder_obs,residual_uses_context=False).to(self.device)
        opt = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.prior_net.parameters()) + list(self.base_decoder.parameters()) + list(self.residual_decoder.parameters()), lr=1e-3, weight_decay=1e-5)
        at, st, ft = tensor(an, self.device), tensor(sn, self.device), tensor(fn, self.device)
        rng = np.random.default_rng(train.get("seed", self.seed_value))
        train_ix,val_ix=validation_split(len(a),train.get("seed",self.seed_value))
        val=torch.as_tensor(val_ix,device=self.device); best=float("inf"); stale=0; patience=int(train.get("early_stopping_patience",3)); self.epochs_run=0
        for epoch in range(self.epochs):
            for local_ix in batch_iter(len(train_ix), self.batch_size, rng):
                ix=train_ix[local_ix]
                ii = torch.as_tensor(ix, device=self.device)
                raw = self.encoder(torch.cat([at[ii], st[ii]], 1))
                q = self._quantize(raw[:, :self.base_dim])
                ids = self._to_ids(q)
                residual_mu,residual_logvar=self._residual_stats(raw[:,self.base_dim:])
                residual=residual_mu+torch.randn_like(residual_mu)*torch.exp(.5*residual_logvar)
                base_pred=self._decode_base(q,ft[ii],st[ii]); residual_pred=self._decode_residual(residual,ft[ii],st[ii],q); pred=base_pred+residual_pred
                pred_zero=self._decode_base(torch.zeros_like(q),ft[ii],st[ii])+self._decode_residual(residual,ft[ii],st[ii],torch.zeros_like(q))
                base_only=base_pred
                prior = self.prior_net(torch.cat([ft[ii], st[ii]], 1))
                logits, r_prior = prior[:, :self.n_codes], prior[:, self.n_codes:]
                kl_r=.5*(residual_mu.square()+residual_logvar.exp()-residual_logvar-1).sum(1).mean()
                recon=(pred-at[ii]).square().mean()
                base_use=torch.relu(self.base_use_margin+recon-(pred_zero-at[ii]).square().mean())
                loss = recon + .25 * F.cross_entropy(logits, ids) + .25 * (r_prior - residual_mu.detach()).square().mean() + self.beta_r*kl_r + self.base_use_weight*(base_use+.5*(base_only-at[ii]).square().mean())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()
            self.epochs_run=epoch+1
            if len(val):
                with torch.no_grad():
                    rawv=self.encoder(torch.cat([at[val],st[val]],1)); qv=self._quantize(rawv[:,:self.base_dim])
                    residual_v,_=self._residual_stats(rawv[:,self.base_dim:])
                    vp=self._decode_base(qv,ft[val],st[val])+self._decode_residual(residual_v,ft[val],st[val],qv)
                    val_loss=(vp-at[val]).square().mean().item()
                best,stale,stop=update_early_stopping(val_loss,best,stale,patience=patience)
                if stop: break
        self.encoder.eval(); self.prior_net.eval(); self.base_decoder.eval(); self.residual_decoder.eval()
        self.tau, self.d_ref = calibrate(train["key"], train["episode_id"], train["init_state_id"], quantile=.95)

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.prior_net.parameters()) + list(self.base_decoder.parameters()) + list(self.residual_decoder.parameters())

    def encode(self, A, s):
        an = norm(np.asarray(A).reshape(len(A), -1), self.am, self.ass)
        sn = norm(s, self.sm, self.ss)
        with torch.no_grad():
            raw = self.encoder(torch.cat([tensor(an, self.device), tensor(sn, self.device)], 1))
            q = self._quantize(raw[:, :self.base_dim])
            ids = self._to_ids(q)
            residual,_=self._residual_stats(raw[:,self.base_dim:])
        return ids.cpu().numpy().astype(np.int64), residual.cpu().numpy()

    def prior(self, f, s):
        fn, sn = norm(f, self.fm, self.fs), norm(s, self.sm, self.ss)
        with torch.no_grad():
            out = self.prior_net(torch.cat([tensor(fn, self.device), tensor(sn, self.device)], 1))
        return out[:, :self.n_codes].cpu().numpy(), out[:, self.n_codes:].cpu().numpy()

    def prior_base(self, base_distribution):
        return np.asarray(base_distribution).argmax(axis=-1).astype(np.int64)

    def _residual_stats(self,x):
        mu,logvar=x.chunk(2,dim=-1); return mu,logvar.clamp(-8,5)

    def residual_kl_bits(self,A,s):
        an=norm(np.asarray(A).reshape(len(A),-1),self.am,self.ass); sn=norm(s,self.sm,self.ss)
        with torch.no_grad():
            raw=self.encoder(torch.cat([tensor(an,self.device),tensor(sn,self.device)],1)); mu,lv=self._residual_stats(raw[:,self.base_dim:]); kl=.5*(mu.square()+lv.exp()-lv-1).sum(1)
        return (kl/np.log(2)).cpu().numpy()

    def base_features(self, b):
        return self._from_ids(b).cpu().numpy()

    def blend_base(self, candidates, weights):
        return np.asarray(weights, dtype=np.float32) @ self.base_features(candidates)

    def decode(self, b, r, f, s):
        bb = np.asarray(b)
        if bb.ndim == 1 and np.issubdtype(bb.dtype, np.integer):
            bb = self.base_features(bb)
        fn, sn = norm(f, self.fm, self.fs), norm(s, self.sm, self.ss)
        with torch.no_grad():
            bt=tensor(bb,self.device); y = (self._decode_base(bt,tensor(fn,self.device),tensor(sn,self.device))+self._decode_residual(tensor(r,self.device),tensor(fn,self.device),tensor(sn,self.device),bt)).cpu().numpy()
        return (y * self.ass + self.am).reshape(len(y), *self.action_shape)

    def _context(self,f,s): return torch.cat([*(([f] if self.decoder_obs else [])),s],1)
    def _decode_base(self,b,f,s): return self.base_decoder(torch.cat([b,self._context(f,s)],1))
    def _decode_residual(self,r,f,s,b):
        return self.residual_decoder(r)

    def memory_add(self, k, b, meta):
        return memory_add(self.memory, k, b, meta, self.tau)

    def memory_query(self, k, meta=None):
        return memory_query(self.memory, k, meta, self.topk, self.temperature, self.d_ref)

    def describe(self):
        return {"name": "FSQ", "base": f"FSQ {self.n_codes}-code product of {self.base_dim} binary scalar levels",
                "residual": f"Gaussian {self.residual_dim}-D bottleneck", "residual_decoder_input": "residual only; proprioception and observations cannot bypass the residual bottleneck", "objective": "normalized action MSE + 0.25 prior CE + residual regression + beta_r*KL(q(r|A,s)||N(0,I)) + base-use margin penalty",
                "beta_r":self.beta_r,"base_use_weight":self.base_use_weight,"base_use_margin":self.base_use_margin,"decoder_obs":self.decoder_obs,"decoder_observation_conditioning":"comparison only; default is false","residual_kl_weights_compared":[1.0,10.0],
                "quantization": "fixed scalar levels {-1,+1}; no learned codebook or EMA", "updates_without_gradient": "none during memory insertion",
                "neural_weights_fixed_on_memory_add": True, "memory_representatives_change_on_add": False, "old_ids_change_on_add": False}


def build_variants():
    variants={}
    for k in (16,32,64):
        variants[f"FSQ_{k}"]=FSQ(n_codes=k,beta_r=1.0)
        variants[f"FSQ_{k}_br10"]=FSQ(n_codes=k,beta_r=10.0)
    variants["FSQ_32_decoder_obs"]=FSQ(n_codes=32,beta_r=1.0,decoder_obs=True)
    return variants
