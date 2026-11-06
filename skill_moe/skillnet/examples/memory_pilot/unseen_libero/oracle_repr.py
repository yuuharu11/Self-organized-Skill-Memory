"""Representation transforms and fixed M6 medoids for unseen-LIBERO oracle study.

The transforms are deliberately independent of query ranking: they encode a
10x7 action window and decode it to a real-valued 10x7 draft.  The oracle
selector is applied later to choose one actual memory window/medoid by target
active-7 MSE; no candidates are averaged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.fft import dct, idct


REPRESENTATIONS = ("raw", "b12", "pca24", "pca12", "dct14", "dct21", "keyframe21", "keyframe35", "zero")
T0_GRID = (.05, .1, .2, .3, .4, .5, .6, .8, 1.0)


def encode_fixed(actions: np.ndarray, name: str, *, pca=None) -> np.ndarray:
    """Encode [N,10,7] active actions into the named fixed representation."""
    x = np.asarray(actions, np.float32)
    if x.ndim != 3 or x.shape[1:] != (10, 7):
        raise ValueError(f"expected [N,10,7], got {x.shape}")
    if name == "raw":
        return x.reshape(len(x), 70).copy()
    if name.startswith("pca"):
        if pca is None:
            raise ValueError("PCA model required")
        return pca.transform(x.reshape(len(x), 70)).astype(np.float32)
    if name.startswith("dct"):
        k = {"dct14": 2, "dct21": 3}[name]
        return dct(x, type=2, axis=1, norm="ortho")[:, :k].reshape(len(x), 7*k).astype(np.float32)
    if name.startswith("keyframe"):
        k = {"keyframe21": 3, "keyframe35": 5}[name]
        ix = np.linspace(0, 9, k).round().astype(np.int64)
        return x[:, ix].reshape(len(x), 7*k).copy()
    raise ValueError(f"encode_fixed is not defined for {name!r}")


def decode_fixed(values: np.ndarray, name: str, *, pca=None) -> np.ndarray:
    """Decode a fixed representation to [N,10,7] active actions."""
    z = np.asarray(values, np.float32)
    if name == "raw":
        if z.ndim != 2 or z.shape[1] != 70: raise ValueError(z.shape)
        return z.reshape(-1, 10, 7).copy()
    if name.startswith("pca"):
        if pca is None: raise ValueError("PCA model required")
        return pca.inverse_transform(z).reshape(-1,10,7).astype(np.float32)
    if name.startswith("dct"):
        k = {"dct14": 2, "dct21": 3}[name]
        if z.shape[1] != 7*k: raise ValueError(z.shape)
        coeff = np.zeros((len(z),10,7),np.float32)
        coeff[:,:k] = z.reshape(len(z),k,7)
        return idct(coeff,type=2,axis=1,norm="ortho").astype(np.float32)
    if name.startswith("keyframe"):
        k = {"keyframe21": 3, "keyframe35": 5}[name]
        if z.shape[1] != 7*k: raise ValueError(z.shape)
        ix = np.linspace(0,9,k).round().astype(np.int64)
        y=z.reshape(len(z),k,7)
        out=np.empty((len(z),10,7),np.float32)
        for n in range(len(z)):
            for d in range(7): out[n,:,d]=np.interp(np.arange(10),ix,y[n,:,d])
        return out
    raise ValueError(f"decode_fixed is not defined for {name!r}")


def active7_mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x,y=np.asarray(a,np.float32),np.asarray(b,np.float32)
    if x.shape != y.shape or x.shape[-2:] != (10,7): raise ValueError((x.shape,y.shape))
    return np.mean((x-y)**2,axis=(-2,-1))


class RepresentationMetric:
    """Euclidean squared distance scaled by representation width."""
    def __init__(self, dim: int): self.dim=int(dim)
    def distance(self,a,b):
        x=np.asarray(a,np.float32).reshape(-1);y=np.asarray(b,np.float32).reshape(-1)
        if x.shape != (self.dim,) or y.shape != (self.dim,): raise ValueError((x.shape,y.shape,self.dim))
        return float(np.dot(x-y,x-y)/self.dim)
    def embeddings(self,values): return np.asarray(values,np.float32)/np.sqrt(self.dim)
    def nearest(self,value,centers):
        if len(centers)==0:return None,float("inf")
        c=np.asarray(centers,np.float32)
        d=np.einsum('ij,ij->i',c-value,c-value)/self.dim
        i=int(np.argmin(d));return i,float(d[i])


def m6_default_medoids(values: np.ndarray, task: np.ndarray, episode: np.ndarray,
                       row_ids: np.ndarray, *, seed=20261104):
    """Fit only the predeclared M6 default; return medoid local and global IDs."""
    from skill_memories.build import m6_fit, unit_medoid
    x=np.asarray(values,np.float32);ids=np.asarray(row_ids,np.int64)
    metric=RepresentationMetric(x.shape[1])
    fit=m6_fit(x,ids,np.asarray(task,np.int32),np.asarray(episode,np.int32),metric,
               kappa=.1,seed=seed,min_support=10,min_episodes=3)
    med_local=np.asarray([unit_medoid(m,x,metric) for m in fit['members_local']],np.int64)
    return {"row_units":np.asarray(fit["row_units"],np.int32),"medoid_local":med_local,
            "medoid_rows":ids[med_local],"unit_count":int(len(med_local)),
            "unassigned_rate":float(np.mean(np.asarray(fit["row_units"])<0)),
            "epsilon":float(fit["epsilon"]),"functional_median":float(fit["functional_median"])}


def oracle_candidate(target: np.ndarray, candidate_ids: np.ndarray, candidate_drafts: np.ndarray):
    """Choose one candidate by true active7 MSE; ties preserve input order."""
    ids=np.asarray(candidate_ids,np.int64); drafts=np.asarray(candidate_drafts,np.float32)
    y=np.asarray(target,np.float32)
    if drafts.shape != (len(ids),10,7): raise ValueError((drafts.shape,len(ids)))
    err=np.mean((drafts-y[None])**2,axis=(-2,-1))
    j=int(np.argmin(err))
    return int(ids[j]),drafts[j].copy(),float(err[j])


def deterministic_cross_task_shuffle(candidate_ids, query_task, candidate_task, *, seed=4101):
    """Return a bijection of a candidate list whose source task differs per query."""
    ids=np.asarray(candidate_ids,np.int64);qt=np.asarray(query_task);ct=np.asarray(candidate_task)[ids]
    if ids.shape != qt.shape: raise ValueError((ids.shape,qt.shape))
    from scipy.optimize import linear_sum_assignment
    rng=np.random.default_rng(seed)
    cost=(qt[:,None]==ct[None,:]).astype(np.float64)*1e6+rng.random((len(ids),len(ids)))*1e-3
    r,c=linear_sum_assignment(cost)
    if not np.array_equal(r,np.arange(len(ids))): raise AssertionError("assignment is not a bijection")
    if np.any(qt == ct[c]): raise ValueError("no cross-task bijection exists")
    return ids[c]


def deterministic_cross_task_from_pool(candidate_pool_ids, query_task, candidate_task, *, seed=4101):
    """Choose unique shuffled pool candidates, excluding each query task."""
    from scipy.optimize import linear_sum_assignment
    ids=np.unique(np.asarray(candidate_pool_ids,np.int64));qt=np.asarray(query_task);ct=np.asarray(candidate_task)[ids]
    if not len(ids):raise ValueError('empty candidate pool')
    rng=np.random.default_rng(seed)
    if len(ids)<len(qt):
        out=np.empty(len(qt),np.int64)
        for i,t in enumerate(qt):
            eligible=ids[ct!=t]
            if not len(eligible):raise ValueError(f'candidate pool has no cross-task rows for task {t}')
            out[i]=rng.choice(eligible)
        return out
    cost=rng.random((len(qt),len(ids)))
    cost[qt[:,None]==ct[None,:]]=1e6
    r,c=linear_sum_assignment(cost)
    if len(r)!=len(qt) or not np.array_equal(r,np.arange(len(qt))) or np.any(qt==ct[c]):
        raise ValueError('candidate pool cannot supply unique cross-task candidates')
    return ids[c]


def episode_bootstrap_diff(a,b,episode,*,reps=5000,seed=4101):
    d=np.asarray(a,np.float64)-np.asarray(b,np.float64);ep=np.asarray(episode)
    if d.shape != ep.shape: raise ValueError((d.shape,ep.shape))
    unique,inverse=np.unique(ep,return_inverse=True)
    sums=np.bincount(inverse,weights=d);counts=np.bincount(inverse)
    rng=np.random.default_rng(seed);ix=rng.integers(0,len(unique),size=(reps,len(unique)))
    boot=sums[ix].sum(1)/counts[ix].sum(1)
    return {"mean":float(d.mean()),"ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
            "episodes":int(len(unique)),"rows":int(len(d)),"reps":int(reps)}
