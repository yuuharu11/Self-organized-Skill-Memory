"""S2 action-space codec and append-only M6 memory helpers.

The supplied training cache must contain only LIBERO-90 examples transformed
with the 90 checkpoint's stored normalization statistics. Arrays use the
policy action convention: 10 future actions, 32 channels (7 LIBERO channels
followed by the normal OpenPI padding), and state width 32.
"""
from __future__ import annotations

import hashlib
import copy
import pickle
from pathlib import Path
import sys
from typing import Any

import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[2]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))


ACTION_SHAPE = (10, 32)
REPRESENTATION_VERSION = "pi05-libero90-normalized-action-10x32-v1"
KEY_TEXT_SOURCES = ("instruction_only", "pi05_generated_subtask")


def validate_key_text_source(data: dict[str, Any]) -> str:
    """Require auditable, non-annotation text provenance for every cached key."""
    if "key_text_source" not in data:
        raise ValueError("key cache must declare key_text_source as instruction_only or pi05_generated_subtask")
    sources = np.asarray(data["key_text_source"]).astype(str).reshape(-1)
    unique = np.unique(sources)
    if len(unique) != 1 or unique[0] not in KEY_TEXT_SOURCES:
        raise ValueError(f"unsupported or mixed key_text_source values: {unique.tolist()}")
    source = str(unique[0])
    if source == "pi05_generated_subtask":
        generator = str(np.asarray(data.get("key_text_generator", "")).item())
        if generator != "pi05":
            raise ValueError("generated-subtask keys must declare key_text_generator='pi05'")
        checkpoint = str(np.asarray(data.get("key_text_generator_checkpoint", "")).item())
        if not checkpoint:
            raise ValueError("generated-subtask keys must declare key_text_generator_checkpoint")
    return source


def _validate_arrays(data: dict[str, Any]) -> None:
    required = ("actions", "state", "key", "episode_id", "init_state_id", "suite", "normalization_sha256")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"S2 memory cache is missing fields: {missing}")
    actions = np.asarray(data["actions"])
    states = np.asarray(data["state"])
    keys = np.asarray(data["key"])
    n = len(actions)
    if actions.shape != (n, *ACTION_SHAPE):
        raise ValueError(f"actions must have [N,{ACTION_SHAPE[0]},{ACTION_SHAPE[1]}], got {actions.shape}")
    if states.ndim != 2 or states.shape != (n, 32):
        raise ValueError(f"state must be [N,32], got {states.shape}")
    if keys.ndim != 2 or keys.shape[0] != n:
        raise ValueError(f"key must be [N,key_dim], got {keys.shape}")
    for name in ("episode_id", "init_state_id"):
        if len(np.asarray(data[name])) != n:
            raise ValueError(f"{name} length does not match actions")
    if not np.all(np.asarray(data["suite"]).astype(str) == "libero90"):
        raise ValueError("M6 fitting accepts LIBERO-90 rows only")
    if not all(np.all(np.isfinite(np.asarray(data[name]))) for name in ("actions", "state", "key")):
        raise ValueError("actions, state, and key must be finite")
    norm_hash = str(np.asarray(data["normalization_sha256"]).item())
    if len(norm_hash) != 64 or any(ch not in "0123456789abcdef" for ch in norm_hash.lower()):
        raise ValueError("normalization_sha256 must be a 64-character SHA-256 digest")
    validate_key_text_source(data)


def fit_memory(data: dict[str, Any], *, seed: int = 0, epochs: int = 10):
    """Fit M1+M6 on the provided LIBERO-90-only rows and seed episodic memory."""
    _validate_arrays(data)
    from emergent.light.methods.m6 import M6

    arrays = {key: np.asarray(data[key]) for key in ("actions", "state", "key", "episode_id", "init_state_id")}
    memory = M6(kappa=0.1, min_support=10, min_episodes=3, encoder="m1", epochs=epochs,
                batch_size=128, seed=seed, base_dim=12, residual_dim=4)
    memory.fit({
        "actions": arrays["actions"], "state": arrays["state"], "feature": arrays["key"],
        "key": arrays["key"], "episode_id": arrays["episode_id"],
        "init_state_id": arrays["init_state_id"], "seed": seed,
    })
    before = backbone_digest(memory)
    for index, (key, base) in enumerate(zip(arrays["key"], memory.encode(arrays["actions"], arrays["state"])[0], strict=True)):
        memory.memory_add(key, base, {
            "episode_id": str(arrays["episode_id"][index]),
            "init_state_id": str(arrays["init_state_id"][index]),
            "task": "libero90", "seed": True,
            "raw_action": np.asarray(arrays["actions"][index], np.float32).copy(),
        })
    after = backbone_digest(memory)
    if before != after:
        raise AssertionError("M6 memory insertion changed encoder/decoder weights")
    memory.representation_version = REPRESENTATION_VERSION
    memory.normalization_sha256 = str(np.asarray(data["normalization_sha256"]).item())
    memory.key_text_source = validate_key_text_source(data)
    if memory.key_text_source == "pi05_generated_subtask":
        memory.key_text_generator = str(np.asarray(data["key_text_generator"]).item())
        memory.key_text_generator_checkpoint = str(
            Path(str(np.asarray(data["key_text_generator_checkpoint"]).item())).resolve()
        )
    memory.base_checkpoint_id = "pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999"
    return memory


def backbone_digest(memory) -> str:
    """Stable hash over M1 encoder/prior/decoder parameters and action stats."""
    digest = hashlib.sha256()
    backbone = memory.backbone
    for name in ("am", "ass", "sm", "ss", "fm", "fs"):
        digest.update(np.ascontiguousarray(getattr(backbone, name)).view(np.uint8))
    for name in ("encoder", "prior_net", "base_decoder", "residual_decoder"):
        for parameter in getattr(backbone, name).state_dict().values():
            digest.update(np.ascontiguousarray(parameter.detach().cpu().numpy()).view(np.uint8))
    return digest.hexdigest()


def save_memory(memory, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        pickle.dump(memory, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_memory(path: str | Path):
    with Path(path).open("rb") as stream:
        memory = pickle.load(stream)
    if getattr(memory, "representation_version", None) != REPRESENTATION_VERSION:
        raise ValueError("memory action representation does not match this S2 model")
    return memory


def _retrieved_components(memory, key: np.ndarray, *, top_k: int, meta=None):
    """Return top-keyed M6 rows with cosine-softmax weights as in the protocol."""
    query = np.asarray(key, np.float32).reshape(-1)
    qnorm = max(float(np.linalg.norm(query)), 1e-8)
    query = query / qnorm
    meta = meta or {}
    if '_prepared_candidates' in meta:
        ids=np.asarray(meta['_prepared_candidates'],np.int64)
        ids=ids[ids>=0][:top_k]
        rows=[memory.memory[int(i)] for i in ids]
        if not rows or any(item.get('meta',{}).get('episode_id')==meta.get('episode_id')
                           or item.get('meta',{}).get('init_state_id')==meta.get('init_state_id') for item in rows):
            raise ValueError('invalid independently prepared candidate set')
        keys=np.stack([item['key'] for item in rows]).astype(np.float32)
        keys/=np.maximum(np.linalg.norm(keys,axis=1,keepdims=True),1e-8)
        logits=(keys@query)/max(float(getattr(memory,'temperature',.07)),1e-8)
        weights=np.exp(logits-logits.max());weights/=weights.sum()
        return rows,weights
    eligible = [item for item in memory.memory
                if item.get("meta", {}).get("episode_id") != meta.get("episode_id")
                and item.get("meta", {}).get("init_state_id") != meta.get("init_state_id")]
    if not eligible:
        raise ValueError("no eligible M6 entries for this query")
    keys = np.stack([np.asarray(item["key"], np.float32).reshape(-1) for item in eligible])
    norms = np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8)
    keys = keys / norms
    distances = np.linalg.norm(keys - query[None], axis=1)
    selected = np.argsort(distances, kind="stable")[:min(top_k, len(eligible))]
    rows = [eligible[int(i)] for i in selected]
    similarities = keys[selected] @ query
    temperature = max(float(getattr(memory, "temperature", 0.07)), 1e-8)
    logits = similarities / temperature
    weights = np.exp(logits - logits.max())
    weights = (weights / weights.sum()).astype(np.float32)
    return rows, weights


def _component_values(memory,unit,base,meta):
    if unit is None:
        return np.asarray(base,np.float32),0.
    groups=getattr(memory,'_s2_preparation_members',None)
    all_members=(groups.get(unit,[]) if groups is not None else
                 [item for item in memory.memory if item.get('unit_id')==unit])
    meta=meta or {}
    cache=getattr(memory,'_s2_preparation_values',None)
    episode,initial=meta.get('episode_id'),meta.get('init_state_id')
    ep_map=getattr(memory,'_s2_preparation_ep_init',{})
    cache_key=(unit,None if initial is not None and ep_map.get(episode)==initial else episode,initial)
    if cache is not None and cache_key in cache:return cache[cache_key]
    members=[item for item in all_members
             if ('episode_id' not in meta or item.get('meta',{}).get('episode_id')!=meta['episode_id'])
             and ('init_state_id' not in meta or item.get('meta',{}).get('init_state_id')!=meta['init_state_id'])]
    if not members:
        raise ValueError('no eligible unit members after episode/init exclusion')
    values=np.stack([item['base'] for item in members]).astype(np.float32)
    center=(values.mean(axis=0) if len(members)!=len(all_members) else np.asarray(memory._centers[unit],np.float32))
    sigma=float(np.sqrt(np.mean((values-center[None])**2)))
    if cache is not None:cache[cache_key]=(center,sigma)
    return center,sigma


def retrieval_confidence(memory, key: np.ndarray, *, meta=None) -> tuple[float, float]:
    """Return max cosine similarity and normalized entropy of retrieved weights."""
    rows, weights = _retrieved_components(memory, key, top_k=5, meta=meta)
    query = np.asarray(key, np.float32).reshape(-1)
    query = query / max(float(np.linalg.norm(query)), 1e-8)
    similarities = []
    for row in rows:
        item_key = np.asarray(row["key"], np.float32).reshape(-1)
        item_key = item_key / max(float(np.linalg.norm(item_key)), 1e-8)
        similarities.append(float(query @ item_key))
    entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1e-12))))
    normalized_entropy = entropy / max(float(np.log(len(weights))), 1e-12) if len(weights) > 1 else 0.0
    return max(similarities), float(np.clip(normalized_entropy, 0.0, 1.0))


def retrieval_records(memory, key: np.ndarray, *, meta=None, top_k: int = 5) -> list[dict[str, Any]]:
    rows, weights = _retrieved_components(memory, key, top_k=top_k, meta=meta)
    output = []
    for row, weight in zip(rows, weights, strict=True):
        item_meta = row.get("meta") or {}
        output.append({"unit_id": row.get("unit_id"), "weight": float(weight),
                       "task": item_meta.get("task"), "suite": item_meta.get("suite"),
                       "episode_id": item_meta.get("episode_id"),
                       "init_state_id": item_meta.get("init_state_id")})
    return output


def query_action_mixture(memory, key: np.ndarray, state: np.ndarray, *, top_k: int = 5,
                         meta: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one decoded action per retrieved M6 component and its weight.

    Component spread is measured in M6 base space (the space in which (★)
    defines sigma), and is returned separately. Call ``sample_memory_anchor``
    to sample b' and decode it; adding this sigma in action space is invalid.
    """
    rows, item_weights = _retrieved_components(memory, key, top_k=top_k, meta=meta)
    query = np.asarray(key, np.float32).reshape(1, -1)
    state = np.asarray(state, np.float32).reshape(1, -1)
    _, residual = memory.prior(query, state)
    candidates, weights, scales, seen, seen_index = [], [], [], set(), {}
    for row, weight in zip(rows, item_weights, strict=True):
        base, raw_unit_id = row["base"], row.get("unit_id")
        unit_id = None if raw_unit_id is None else int(raw_unit_id)
        group_key = ("entry", len(candidates)) if unit_id is None else ("unit", unit_id)
        if group_key in seen:
            weights[seen_index[group_key]] += float(weight)
            continue
        seen.add(group_key)
        seen_index[group_key] = len(candidates)
        center,sigma=_component_values(memory,unit_id,base,meta)
        decoded = memory.decode(center[None], residual, query, state)[0]
        candidates.append(np.asarray(decoded, np.float32).reshape(ACTION_SHAPE))
        weights.append(float(weight))
        scales.append(sigma)
    weights = np.asarray(weights, np.float32)
    weights /= max(float(weights.sum()), 1e-12)
    return np.asarray(candidates, np.float32), weights, np.asarray(scales, np.float32)


def sample_memory_anchor(memory, key: np.ndarray, state: np.ndarray, *, seed: int,
                         meta: dict[str, Any] | None = None, return_details: bool = False):
    """Sample a M6 component and b'~N(c_k,sigma_k^2 I), then decode in M1 space."""
    rows, item_weights = _retrieved_components(memory, key, top_k=5, meta=meta)
    # Aggregate retrieved episodic-item weights by M6 unit; pending entries
    # without a unit remain separate candidates.
    groups: dict[tuple[str, int], dict[str, Any]] = {}
    for index, (row, weight) in enumerate(zip(rows, item_weights, strict=True)):
        base, raw_unit = row["base"], row.get("unit_id")
        unit = None if raw_unit is None else int(raw_unit)
        group = ("entry", index) if unit is None else ("unit", unit)
        if group not in groups:
            center,sigma=_component_values(memory,unit,base,meta)
            groups[group] = {"center": center, "sigma": sigma, "weight": 0.0,
                             "unit_id": unit, "task": (row.get("meta") or {}).get("task"),
                             "suite": (row.get("meta") or {}).get("suite")}
        groups[group]["weight"] += float(weight)
    candidates = list(groups.values())
    probabilities = np.asarray([item["weight"] for item in candidates], np.float64)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)
    selected = candidates[int(rng.choice(len(candidates), p=probabilities))]
    base_sample = selected["center"] + selected["sigma"] * rng.normal(size=selected["center"].shape)
    query = np.asarray(key, np.float32).reshape(1, -1)
    state = np.asarray(state, np.float32).reshape(1, -1)
    _, residual = memory.prior(query, state)
    action = memory.decode(base_sample[None], residual, query, state)[0]
    action = np.asarray(action, np.float32).reshape(ACTION_SHAPE)
    details = {"unit_id": selected["unit_id"], "task": selected["task"], "suite": selected["suite"],
               "weight": float(selected["weight"]), "base_sigma": float(selected["sigma"])}
    return (action, details) if return_details else action


def query_raw_action(memory, key: np.ndarray, *, meta=None, return_details: bool = False):
    """RTS-style nearest raw chunk retrieval; uses no success critic."""
    rows, weights = _retrieved_components(memory, key, top_k=1, meta=meta)
    action = (rows[0].get("meta") or {}).get("raw_action")
    if action is None:
        raise ValueError("memory row lacks the original normalized action chunk required by RTS control")
    action = np.asarray(action, np.float32).reshape(ACTION_SHAPE)
    details = {"unit_id": rows[0].get("unit_id"), "task": (rows[0].get("meta") or {}).get("task"),
               "suite": (rows[0].get("meta") or {}).get("suite"), "weight": float(weights[0])}
    return (action, details) if return_details else (action, float(weights[0]))


def sample_optimus_anchor(memory, key: np.ndarray, state: np.ndarray, *, seed: int,
                          meta=None) -> np.ndarray:
    """OptimusVLA-style moment match of retrieved decoded anchors to one Gaussian."""
    candidates, weights, _ = query_action_mixture(memory, key, state, meta=meta)
    mean = np.einsum("k,khd->hd", weights, candidates)
    centered = candidates - mean[None]
    variance = np.einsum("k,khd->hd", weights, centered * centered)
    rng = np.random.default_rng(seed)
    return (mean + np.sqrt(np.maximum(variance, 1e-8)) * rng.normal(size=mean.shape)).astype(np.float32)


def shuffle_memory_values(memory, *, seed: int):
    """Copy memory and permute key-to-value assignments for the shuffle control."""
    shuffled = copy.deepcopy(memory)
    if len(shuffled.memory) < 2:
        return shuffled
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(shuffled.memory))
    values = [(item.get("unit_id"), np.asarray(item["base"]).copy(),
               copy.deepcopy((item.get("meta") or {}).get("raw_action")))
              for item in shuffled.memory]
    for index, source in enumerate(permutation):
        unit_id, base, raw_action = values[int(source)]
        shuffled.memory[index]["unit_id"] = unit_id
        shuffled.memory[index]["base"] = base
        if raw_action is not None:
            shuffled.memory[index].setdefault("meta", {})["raw_action"] = raw_action
    # Invalidate M6's materialized retrieval index after replacing entries.
    shuffled._memory_index_list_id = None
    shuffled._memory_index_keys = shuffled._memory_index_episodes = shuffled._memory_index_initials = None
    shuffled._memory_index_size = 0
    return shuffled
