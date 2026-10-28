#!/usr/bin/env python3
"""Prepare ordinary, retrieved-anchor, and small oracle-anchor training rows."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from memory import load_memory, sample_memory_anchor, validate_key_text_source


def training_neighbors(memory,queries,episodes,initials,*,top_k=5,block=128):
    """Batch the same cosine search with episode/init exclusion, stable ties.

    Avoid rebuilding a N×2048 key matrix once per training row. No model or
    memory value changes; only the independent candidate IDs are prepared.
    """
    keys=np.stack([row['key'] for row in memory.memory]).astype(np.float32)
    keys/=np.maximum(np.linalg.norm(keys,axis=1,keepdims=True),1e-8)
    query=np.asarray(queries,np.float32)
    query=query/np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-8)
    source_eps=np.asarray([row['meta']['episode_id'] for row in memory.memory]).astype(str)
    source_init=np.asarray([row['meta']['init_state_id'] for row in memory.memory]).astype(str)
    _,ep_code=np.unique(np.concatenate([source_eps,np.asarray(episodes).astype(str)]),return_inverse=True)
    _,init_code=np.unique(np.concatenate([source_init,np.asarray(initials).astype(str)]),return_inverse=True)
    n=len(keys);result=np.full((len(query),top_k),-1,np.int64)
    for lo in range(0,len(query),block):
        hi=min(lo+block,len(query))
        scores=query[lo:hi]@keys.T
        invalid=((ep_code[n+lo:n+hi,None]==ep_code[None,:n])
                 |(init_code[n+lo:n+hi,None]==init_code[None,:n]))
        scores[invalid]=-np.inf
        for local,score in enumerate(scores):
            k=min(top_k,int(np.isfinite(score).sum()))
            if not k:continue
            threshold=np.partition(score,len(score)-k)[len(score)-k]
            greater=np.flatnonzero(score>threshold)
            tied=np.flatnonzero(score==threshold)[:k-len(greater)]
            chosen=np.concatenate([greater,tied])
            chosen=chosen[np.lexsort((chosen,-score[chosen]))]
            result[lo+local,:k]=chosen
        if lo%(block*10)==0:print(f'independent candidates: {hi}/{len(query)}',flush=True)
    return result


def prepare(data, memory, *, seed: int = 0):
    required = ("actions", "state", "key", "episode_id", "init_state_id", "suite")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"source cache missing fields: {missing}")
    if not np.all(np.asarray(data["suite"]).astype(str) == "libero90"):
        raise ValueError("source preparation is restricted to LIBERO-90")
    if str(np.asarray(data["normalization_sha256"]).item()) != memory.normalization_sha256:
        raise ValueError("source cache normalization differs from M6 codec normalization")
    if validate_key_text_source(data) != memory.key_text_source:
        raise ValueError("source cache key text differs from the M6 memory key source")
    actions = np.asarray(data["actions"], np.float32)
    states = np.asarray(data["state"], np.float32)
    keys = np.asarray(data["key"], np.float32)
    episodes = np.asarray(data["episode_id"]).astype(str)
    initials = np.asarray(data["init_state_id"]).astype(str)
    bases, _ = memory.encode(actions, states)
    _, prior_residual = memory.prior(keys, states)
    rng = np.random.default_rng(seed)
    neighbors=training_neighbors(memory,keys,episodes,initials)
    # These caches exist only while preparing immutable training sources, and
    # are never persisted or reused after later memory additions.
    from collections import defaultdict
    memory._s2_preparation_members=defaultdict(list)
    ep_init=defaultdict(set)
    for item in memory.memory:
        memory._s2_preparation_members[item.get('unit_id')].append(item)
        ep_init[item['meta']['episode_id']].add(item['meta']['init_state_id'])
    memory._s2_preparation_ep_init={ep:next(iter(ids)) for ep,ids in ep_init.items() if len(ids)==1}
    memory._s2_preparation_values={}
    anchor_actions = np.empty_like(actions)
    source_kind = np.empty((len(actions),), dtype="U16")
    for i in range(len(actions)):
        slot = i % 15
        if slot < 5:
            anchor_actions[i] = actions[i]
            source_kind[i] = "normal"
        elif slot == 14:
            anchor = memory.decode(bases[i:i + 1], prior_residual[i:i + 1],
                                   keys[i:i + 1], states[i:i + 1])[0].reshape(actions.shape[1:])
            anchor_actions[i] = anchor
            source_kind[i] = "target_anchor"
        else:
            meta = {"episode_id": episodes[i], "init_state_id": initials[i],
                    '_prepared_candidates':neighbors[i]}
            try:
                anchor_actions[i] = sample_memory_anchor(
                    memory, keys[i], states[i], seed=int(rng.integers(2**31)), meta=meta
                )
                source_kind[i] = "memory_anchor"
            except ValueError as error:
                raise ValueError(f"independent retrieved anchor unavailable for row {i}; exclusions were not relaxed") from error
        if i%5000==0:print(f'anchor sources: {i}/{len(actions)}',flush=True)
    for name in ('_s2_preparation_members','_s2_preparation_ep_init','_s2_preparation_values'):
        delattr(memory,name)
    return {**data, "anchor_actions": anchor_actions, "source_kind": source_kind,
            "suite": np.asarray(data["suite"]).astype(str)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero90-npz", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    source = Path(args.libero90_npz)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=True) as archive:
        data = {name: archive[name] for name in archive.files}
    result = prepare(data, load_memory(args.memory), seed=args.seed)
    np.savez_compressed(output, **result)
    print({"output": str(output.resolve()), "rows": len(result["actions"]),
           "source_counts": {name: int(np.sum(result["source_kind"] == name))
                             for name in np.unique(result["source_kind"])}})


if __name__ == "__main__":
    main()
