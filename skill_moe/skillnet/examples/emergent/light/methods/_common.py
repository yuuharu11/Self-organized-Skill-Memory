"""Small shared numerical and episodic-memory helpers for method modules."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import hashlib

_CALIBRATION_CACHE = {}


def validation_split(n, seed, fraction=0.1):
    """Deterministic row holdout used only to stop training early."""
    if n < 10: return np.arange(n,dtype=np.int64),np.empty(0,dtype=np.int64)
    rng=np.random.default_rng(int(seed)+413)
    order=rng.permutation(n); n_val=max(1,min(n-1,int(round(n*fraction))))
    return np.sort(order[n_val:]),np.sort(order[:n_val])


def update_early_stopping(metric,best,stale,*,min_delta=1e-4,patience=3):
    metric=float(metric)
    if np.isfinite(metric) and metric < best-min_delta: return metric,0,False
    stale+=1
    return best,stale,stale>=int(patience)


class AdditiveActionDecoder(nn.Module):
    """Differentiable legacy facade for separate base and residual branches."""
    def __init__(self, base_decoder, residual_decoder, *, base_dim, residual_dim,
                 state_dim, feature_dim=0, decoder_obs=False, residual_uses_context=True):
        super().__init__()
        self.base_decoder, self.residual_decoder = base_decoder, residual_decoder
        self.base_dim, self.residual_dim = int(base_dim), int(residual_dim)
        self.state_dim, self.feature_dim = int(state_dim), int(feature_dim)
        self.decoder_obs = bool(decoder_obs)
        self.residual_uses_context = bool(residual_uses_context)

    def forward(self, x):
        i = 0
        base = x[:, i:i+self.base_dim]; i += self.base_dim
        residual = x[:, i:i+self.residual_dim]; i += self.residual_dim
        feature = x[:, i:i+self.feature_dim] if self.decoder_obs else x.new_zeros((len(x), 0))
        if self.decoder_obs: i += self.feature_dim
        state = x[:, i:i+self.state_dim]
        context = torch.cat([feature, state], dim=-1)
        pred = self.base_decoder(torch.cat([base, context], dim=-1))
        residual_input = torch.cat([residual, context], dim=-1) if self.residual_uses_context else residual
        residual_pred = self.residual_decoder(residual_input)
        return pred + residual_pred

    def __getitem__(self, index):
        return self.base_decoder[index]


def seed(seed: int, device: str = "cpu"):
    np.random.seed(seed); torch.manual_seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)


def stats(x):
    x=np.asarray(x,dtype=np.float32); m=x.mean(0); s=x.std(0); s[s<1e-6]=1
    return m,s


def norm(x,m,s): return ((np.asarray(x,dtype=np.float32)-m)/s).astype(np.float32)


def calibrate(keys, episode_id, init_state_id, quantile=.9, block=256, max_rows=4096):
    keys=np.asarray(keys,dtype=np.float32); ep=np.asarray(episode_id).astype(str); st=np.asarray(init_state_id).astype(str)
    if len(keys)<2: return None,None
    digest=hashlib.blake2b(digest_size=16)
    digest.update(keys.tobytes()); digest.update(ep.tobytes()); digest.update(st.tobytes())
    cache_key=(digest.digest(),float(quantile),int(max_rows))
    if cache_key in _CALIBRATION_CACHE: return _CALIBRATION_CACHE[cache_key]
    if len(keys)>max_rows:
        # Deterministic uniform subsample estimates the same within-split
        # nearest-neighbour distance quantiles at O(max_rows^2) cost.
        rng=np.random.default_rng(7081)
        ix=np.sort(rng.choice(len(keys),size=max_rows,replace=False))
        keys,ep,st=keys[ix],ep[ix],st[ix]
    sq=(keys*keys).sum(1); nn=np.full(len(keys),np.inf)
    for lo in range(0,len(keys),block):
        hi=min(len(keys),lo+block)
        d2=np.maximum(sq[lo:hi,None]+sq[None]-2*keys[lo:hi]@keys.T,0)
        bad=(ep[lo:hi,None]==ep[None])|(st[lo:hi,None]==st[None]); d2[bad]=np.inf
        nn[lo:hi]=np.sqrt(d2.min(1))
    nn=nn[np.isfinite(nn)]
    result=((float(np.quantile(nn,quantile)),float(np.median(nn))) if len(nn) else (None,None))
    _CALIBRATION_CACHE[cache_key]=result
    if len(_CALIBRATION_CACHE)>64: _CALIBRATION_CACHE.pop(next(iter(_CALIBRATION_CACHE)))
    return result


def memory_add(memory, key, base, meta, tau):
    key=np.asarray(key,dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(key)) or not np.all(np.isfinite(np.asarray(base,dtype=float))): return False,"nonfinite"
    if meta.get("seed",False):
        memory.append({"key":key.copy(),"base":np.array(base,copy=True),"meta":dict(meta)})
        return True,"seed"
    if not memory:
        memory.append({"key":key.copy(),"base":np.array(base,copy=True),"meta":dict(meta)})
        return True,"added_empty"
    eligible=[e for e in memory if e["meta"].get("episode_id")!=meta.get("episode_id")
              and e["meta"].get("init_state_id")!=meta.get("init_state_id")]
    d=min((float(np.linalg.norm(key-e["key"])) for e in eligible),default=np.inf)
    if tau is not None and d<=tau: return False,"near_existing"
    memory.append({"key":key.copy(),"base":np.array(base,copy=True),"meta":dict(meta)})
    return True,"added_novel"


def memory_query(memory,key,meta=None,k=5,temperature=.5,d_ref=None):
    meta=meta or {}; key=np.asarray(key,dtype=np.float32).reshape(-1)
    eligible=[e for e in memory if e["meta"].get("episode_id")!=meta.get("episode_id")
              and e["meta"].get("init_state_id")!=meta.get("init_state_id")]
    if not eligible: return np.zeros(0,dtype=np.float32),np.zeros(0,dtype=np.float32),[]
    ds=np.asarray([np.linalg.norm(key-e["key"]) for e in eligible]); order=np.argsort(ds,kind="stable")[:min(k,len(ds))]
    chosen=[eligible[i] for i in order]; d=ds[order]
    scale=max(float(temperature)*(float(d_ref) if d_ref else max(float(d[0]),1e-12)),1e-12)
    w=np.exp(-(d-d[0])/scale); w=w/w.sum()
    bases=np.stack([np.asarray(e["base"]) for e in chosen])
    return bases.astype(np.float32),w.astype(np.float32),chosen


def tensor(x,device): return torch.as_tensor(np.asarray(x,dtype=np.float32),device=device)


def batch_iter(n,batch_size,rng):
    order=rng.permutation(n)
    for i in range(0,n,batch_size): yield order[i:i+batch_size]
