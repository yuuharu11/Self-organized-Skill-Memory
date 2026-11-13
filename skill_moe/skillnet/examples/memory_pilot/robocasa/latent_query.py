"""RoboCasa latent-token query (one learned Z0 cross-attending to π0.5 VLM tokens).

Stages are deliberately explicit: extract token cache, fit/evaluate three
episode-split seeds, and validate the exact serving prefix path against cache.
The frozen π0.5 is used only for VLM prefix tokens; it does not sample actions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import context_key as ctx
import experiment as ex
import expert

STATE = Path(os.environ.get("SKILLNET_STATE_DIR", "/work/.skillnet-state"))
ROOT = Path(os.environ.get("ROBOCASA_OUTPUT_ROOT", STATE / "outputs/robocasa-memory"))
CTX = Path(os.environ.get("ROBOCASA_CTX_OUT", ROOT / "ctx-01"))
SOURCE = Path(os.environ.get("ROBOCASA_MEMORY_OUT", ROOT / "memory-01/policy-order-v2"))
GT = Path(os.environ.get("ROBOCASA_GTDEF2", ROOT / "gt-def2"))
OUT = Path(os.environ.get("ROBOCASA_LATENT_OUT", ROOT / "latent-query-01"))
TOKEN_DIR = OUT / "tokens"
SEEDS = (4101, 4102, 4103)
WIDTH = 256
TOKEN_DIM = 2048
HISTORY = 10


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=ex._json_default) + "\n")
    tmp.replace(path)


def _progress(stage, **kwargs):
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage, **kwargs}
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def _refuse_existing(path):
    if Path(path).exists():
        raise FileExistsError(f"refusing to overwrite existing result: {path}")


def _id(task, episode, frame):
    return f"{int(task)}:{int(episode)}:{int(frame)}"


class _ShardedNpy:
    """Read-only virtual concatenation used to resume large token caches."""
    def __init__(self, prefix, suffix, prefix_count):
        self.prefix = prefix
        self.suffix = suffix
        self.prefix_count = int(prefix_count)
        self.shape = (self.prefix_count + len(suffix), *suffix.shape[1:])
        self.dtype = suffix.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            first, *rest = key
            value = self[first]
            return value[tuple(rest)] if rest else value
        indices = np.arange(self.shape[0])[key]
        if np.isscalar(indices):
            i = int(indices)
            return self.prefix[i] if i < self.prefix_count else self.suffix[i-self.prefix_count]
        indices = np.asarray(indices)
        result = np.empty((*indices.shape, *self.shape[1:]), dtype=self.dtype)
        from_prefix = indices < self.prefix_count
        if np.any(from_prefix):
            result[from_prefix] = self.prefix[indices[from_prefix]]
        if np.any(~from_prefix):
            result[~from_prefix] = self.suffix[indices[~from_prefix]-self.prefix_count]
        return result


def _load_token_arrays():
    """Load full or prefix+tail-sharded token cache with ndarray indexing."""
    manifest = json.loads((TOKEN_DIR / "manifest.json").read_text())
    if manifest.get("resume_rows", 0):
        source = Path(manifest["resume_source"]) / "tokens"
        count = int(manifest["resume_rows"])
        tok = _ShardedNpy(np.load(source / "tokens_fp16.npy", mmap_mode="r"),
                          np.load(TOKEN_DIR / "tokens_tail_fp16.npy", mmap_mode="r"), count)
        mask = _ShardedNpy(np.load(source / "valid.npy", mmap_mode="r"),
                           np.load(TOKEN_DIR / "valid_tail.npy", mmap_mode="r"), count)
        return tok, mask
    return (np.load(TOKEN_DIR / "tokens_fp16.npy", mmap_mode="r"),
            np.load(TOKEN_DIR / "valid.npy", mmap_mode="r"))


def _context_inventory():
    """Regenerate ctx-01's deterministic row order and append gt-def2 queries."""
    manifest, pool, records, candidates = ctx._inventory()
    contexts = {}
    for split in ("memory", "heldout"):
        for r in records[split]:
            contexts.setdefault(_id(r["task"], r["episode"], r["frame"]), r)
    for r in candidates:
        contexts.setdefault(_id(r["task"], r["episode"], r["frame"]), r)
    gt = np.load(GT / "query_features.npz", allow_pickle=False)
    per_task = {int(t): x for t, x in enumerate(manifest["task_splits"])}
    episode_cache = {}
    gt_ids = []
    for task, episode, frame in zip(gt["task"], gt["episode"], gt["frame"], strict=True):
        key = _id(task, episode, frame)
        if key not in contexts:
            pair = (int(task), int(episode))
            if pair not in episode_cache:
                episode_cache[pair] = ex._load_cached_episode(SOURCE, *pair)
            actions, states = episode_cache[pair][:2]
            info = next(d for d in manifest["demos"] if int(d["task_id"]) == pair[0] and int(d["episode"]) == pair[1])
            task_info = per_task[pair[0]]
            contexts[key] = {"task": pair[0], "episode": pair[1], "global_episode": int(info["global_episode"]),
                             "frame": int(frame), "state": states[int(frame)].copy(), "hist": np.zeros((10, 12), np.float32),
                             "target": actions[int(frame):int(frame)+10].copy(), "prompt": str(info["prompt"]),
                             "root": str(task_info["root"])}
        gt_ids.append(key)
    return manifest, pool, records, candidates, list(contexts.values()), gt_ids


def _raw_observation(row):
    images = [ctx._read_camera(row["root"], cam, row["episode"], row["frame"]) for cam in ctx.CAMERAS]
    return {"observation/image": images[0], "observation/right_image": images[1],
            "observation/wrist_image": images[2],
            "observation/state": ex.state_storage_to_policy(row["state"]), "prompt": row["prompt"]}


def make_prefix_token_extractor(model):
    """Return the server/offline-identical final VLM prefix token extractor."""
    import flax.nnx as nnx
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask
    from openpi.shared.nnx_utils import module_jit

    class PrefixTokens(nnx.Module):
        def __init__(self, base):
            self.base = base

        def __call__(self, observation):
            obs = model_api.preprocess_observation(None, observation, train=False)
            tokens, valid, ar = self.base.embed_prefix(obs)
            positions = jnp.cumsum(valid, axis=1) - 1
            (hidden, _), _ = self.base.PaliGemma.llm(
                [tokens, None], mask=make_attn_mask(valid, ar), positions=positions)
            weights = valid.astype(hidden.dtype)[..., None]
            pooled = jnp.sum(hidden * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1)
            return pooled.astype(jnp.float16), hidden.astype(jnp.float16), valid

    return module_jit(PrefixTokens(model).__call__)


def _group_ids(num_images=3, image_tokens=256, text_tokens=200):
    """Camera-group IDs follow left/right/wrist transform order; 3 denotes text."""
    return np.concatenate([np.repeat(np.arange(num_images, dtype=np.int8), image_tokens),
                           np.full(text_tokens, num_images, np.int8)])


def extract_tokens(*, batch_size=4, max_contexts=None, resume_from=None, resume_rows=None):
    import jax
    import jax.numpy as jnp
    import torch
    from openpi.models import model as model_api

    if TOKEN_DIR.exists() and any(TOKEN_DIR.iterdir()):
        raise FileExistsError(f"refusing to overwrite prefix-token cache: {TOKEN_DIR}")
    if torch.cuda.is_available():
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
        torch.cuda.set_per_process_memory_fraction(.20)
    _manifest, _pool, records, candidates, contexts, gt_ids = _context_inventory()
    if max_contexts is not None:
        contexts = contexts[:int(max_contexts)]
        if not gt_ids or any(k not in {_id(r["task"],r["episode"],r["frame"]) for r in contexts} for k in gt_ids):
            raise ValueError("--max-contexts smoke truncation does not include all evaluation rows")
    transform = ctx._make_transforms(expert.CHECKPOINT)
    model, _ = expert._load_expert_model(expert.CHECKPOINT)
    extract = make_prefix_token_extractor(model)
    first = contexts[:min(int(batch_size), len(contexts))]
    items = [transform(_raw_observation(r)) for r in first]
    batched = jax.tree.map(lambda *xs: jnp.asarray(np.stack([np.asarray(x) for x in xs])), *items)
    obs = model_api.Observation.from_dict(batched)
    pooled, token_batch, mask_batch = extract(obs)
    token_shape = tuple(token_batch.shape[1:])
    if token_shape[-1] != TOKEN_DIM:
        raise ValueError(f"expected final VLM width {TOKEN_DIM}, got {token_shape}")
    seq = int(token_shape[0])
    feature_table = np.asarray(np.load(CTX / "cache/prefix_mean_fp16.npy", mmap_mode="r"), np.float16)
    # The ctx cache index order is precisely the memory->heldout->candidate context order.
    expected_context_count = int(json.loads((CTX / "cache/feature_manifest.json").read_text())["contexts"])
    if len(contexts) < expected_context_count:
        raise AssertionError(f"reconstructed only {len(contexts)} ctx contexts, expected >= {expected_context_count}")
    # ctx contexts occupy the prefix of the regenerated order. Verify mean before fp16 token rounding.
    mean_tolerance = 0.05
    def compare_mean(pooled_chunk, start, size):
        expected = feature_table[start:start+size]
        actual = np.asarray(pooled_chunk[:size], np.float16)
        diff = np.abs(actual.astype(np.float32)-expected.astype(np.float32))
        if diff.size and float(diff.max()) > mean_tolerance:
            raise AssertionError(f"prefix mean does not reproduce ctx cache: max abs {diff.max():.6g} > {mean_tolerance}")
        return float(diff.max()) if diff.size else 0.0
    maximum_mean_error = compare_mean(pooled, 0, len(first))
    TOKEN_DIR.mkdir(parents=True, exist_ok=False)
    resume_count = 0
    resume_source = None
    if resume_from is not None:
        if resume_rows is None or int(resume_rows) <= 0:
            raise ValueError("--resume-rows must be positive when --resume-from is used")
        source_dir = Path(resume_from) / "tokens"
        source_tokens = np.load(source_dir / "tokens_fp16.npy", mmap_mode="r")
        source_masks = np.load(source_dir / "valid.npy", mmap_mode="r")
        resume_count = int(resume_rows)
        if resume_count > len(contexts) or source_tokens.shape[1:] != (seq, TOKEN_DIM) or source_masks.shape[1:] != (seq,):
            raise ValueError("resume token cache shape does not match this extraction")
        if source_tokens.shape[0] < resume_count or source_masks.shape[0] < resume_count:
            raise ValueError("resume source is shorter than --resume-rows")
        resume_source = str(Path(resume_from).resolve())
        tokens = np.lib.format.open_memmap(TOKEN_DIR / "tokens_tail_fp16.npy", mode="w+", dtype=np.float16,
                                           shape=(len(contexts)-resume_count, seq, TOKEN_DIM))
        masks = np.lib.format.open_memmap(TOKEN_DIR / "valid_tail.npy", mode="w+", dtype=np.bool_,
                                          shape=(len(contexts)-resume_count, seq))
        _progress("resume_prefix_referenced", resume_rows=resume_count, source=resume_source)
    else:
        tokens = np.lib.format.open_memmap(TOKEN_DIR / "tokens_fp16.npy", mode="w+", dtype=np.float16,
                                           shape=(len(contexts), seq, TOKEN_DIM))
        masks = np.lib.format.open_memmap(TOKEN_DIR / "valid.npy", mode="w+", dtype=np.bool_,
                                          shape=(len(contexts), seq))
    token_map = {_id(r["task"], r["episode"], r["frame"]): i for i, r in enumerate(contexts)}
    max_batches = (len(contexts) - resume_count + int(batch_size) - 1) // int(batch_size)
    for bi, start in enumerate(range(resume_count, len(contexts), int(batch_size))):
        chunk = contexts[start:start+int(batch_size)]
        if start == 0:
            pp, tt, mm = pooled, token_batch, mask_batch
        else:
            chunk_items = [transform(_raw_observation(r)) for r in chunk]
            bb = jax.tree.map(lambda *xs: jnp.asarray(np.stack([np.asarray(x) for x in xs])), *chunk_items)
            pp, tt, mm = extract(model_api.Observation.from_dict(bb))
        if start < expected_context_count:
            maximum_mean_error = max(maximum_mean_error, compare_mean(pp, start, min(len(chunk), expected_context_count-start)))
        write_start = start - resume_count
        tokens[write_start:write_start+len(chunk)] = np.asarray(tt, np.float16)
        masks[write_start:write_start+len(chunk)] = np.asarray(mm, bool)
        if bi % 100 == 0 or bi + 1 == max_batches:
            _progress("extract_tokens", contexts_done=min(start+len(chunk),len(contexts)), contexts=len(contexts),
                      batch=bi+1, batches=max_batches, maximum_pooled_mean_abs_error=maximum_mean_error)
    tokens.flush(); masks.flush()
    # Infer image token count from transformed observation, retaining explicit group IDs.
    image_tokens_total = seq - int(obs.tokenized_prompt.shape[1])
    if image_tokens_total % len(ctx.CAMERAS):
        raise AssertionError(f"prefix layout does not split into {len(ctx.CAMERAS)} cameras and text: {seq=}")
    image_per_camera = image_tokens_total // len(ctx.CAMERAS)
    groups = np.concatenate([np.repeat(np.arange(len(ctx.CAMERAS), dtype=np.int8), image_per_camera),
                             np.full(int(obs.tokenized_prompt.shape[1]), len(ctx.CAMERAS), np.int8)])
    np.save(TOKEN_DIR / "group_ids.npy", groups)
    np.savez_compressed(TOKEN_DIR / "index.npz",
        task=np.asarray([r["task"] for r in contexts], np.int16),
        episode=np.asarray([r["episode"] for r in contexts], np.int32),
        frame=np.asarray([r["frame"] for r in contexts], np.int32),
        memory_context=np.asarray(np.load(CTX/"cache/rows.npz")["memory_context"],np.int32),
        heldout_context=np.asarray(np.load(CTX/"cache/rows.npz")["heldout_context"],np.int32),
        candidate_context=np.asarray(np.load(CTX/"cache/rows.npz")["candidate_context"],np.int32),
        gt_query_context=np.asarray([token_map[k] for k in gt_ids],np.int32),
        gt_task=np.asarray(np.load(GT/"query_features.npz")["task"],np.int16),
        gt_episode=np.asarray(np.load(GT/"query_features.npz")["episode"],np.int32),
        gt_frame=np.asarray(np.load(GT/"query_features.npz")["frame"],np.int32))
    manifest = {"checkpoint": str(expert.CHECKPOINT), "source": str(SOURCE), "context_cache": str(CTX),
        "gtdef2": str(GT), "n_contexts": len(contexts), "sequence_length": seq, "token_dim": TOKEN_DIM,
        "dtype": "float16", "validity_mask": "one bool per prefix token", "group_ids":
        {str(i): name for i,name in enumerate((*ctx.CAMERAS,"text"))},
        "camera_order": list(ctx.CAMERAS), "hidden_layer": "final PaliGemma transformer output before action expert",
        "ctx_mean_max_abs_error": maximum_mean_error, "ctx_mean_tolerance": mean_tolerance,
        "token_cache_bytes": int(tokens.nbytes),
        "history_query_dim": int(2048+16+10*12+10), "ctx_feature_index_order_preserved": True,
        "resume_rows": resume_count, "resume_source": resume_source,
        "token_shards": (["tokens_tail_fp16.npy"] if resume_count else ["tokens_fp16.npy"]),
        "mask_shards": (["valid_tail.npy"] if resume_count else ["valid.npy"])}
    _write_json(TOKEN_DIR / "manifest.json", manifest)
    _progress("extract_complete", **manifest)


def _network_classes():
    import torch
    from torch import nn
    class LatentQuery(nn.Module):
        def __init__(self, query_dim, candidate_dim=60, token_dim=2048, d_model=256, heads=8):
            super().__init__()
            self.z0 = nn.Parameter(torch.randn(1, 1, d_model) * .02)
            self.token_kv = nn.Linear(token_dim, d_model)
            self.cross_attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
            self.attn_norm = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
            self.ffn_norm = nn.LayerNorm(d_model)
            self.history_state = nn.Sequential(nn.Linear(query_dim, 512), nn.GELU(), nn.Linear(512, d_model), nn.GELU())
            self.query = nn.Sequential(nn.Linear(2*d_model, 512), nn.GELU(), nn.Linear(512, d_model))
            self.key = nn.Sequential(nn.Linear(candidate_dim, 512), nn.GELU(), nn.Linear(512, d_model))
            self.log_scale = nn.Parameter(torch.tensor(float(np.log(20.))))
        def encode_query(self, tokens, token_mask, query_input):
            kv = self.token_kv(tokens)
            q0 = self.z0.expand(len(tokens), -1, -1)
            attended, _ = self.cross_attention(q0, kv, kv, key_padding_mask=~token_mask.bool(), need_weights=False)
            latent0 = self.attn_norm(q0 + attended)
            latent = self.ffn_norm(latent0 + self.ffn(latent0))[:, 0]
            h = self.history_state(query_input)
            return torch.nn.functional.normalize(self.query(torch.cat((latent,h),dim=-1)), dim=-1)
        def forward(self, tokens, token_mask, query_input, candidate_types):
            query = self.encode_query(tokens, token_mask, query_input)
            keys = torch.nn.functional.normalize(self.key(candidate_types), dim=-1)
            return query @ keys.T * self.log_scale.exp()
    return LatentQuery


def _episode_folds(task, episode, seed, nfolds=5):
    pairs = sorted(set(zip(np.asarray(task).astype(int).tolist(), np.asarray(episode).astype(int).tolist())))
    by_task = {}
    for t,e in pairs: by_task.setdefault(t,[]).append(e)
    rng = np.random.default_rng(int(seed)); assignment={}
    for t, eps in sorted(by_task.items()):
        perm = np.asarray(eps)[rng.permutation(len(eps))]
        for j,e in enumerate(perm): assignment[(t,int(e))]=int(j % nfolds)
    return np.asarray([assignment[(int(t),int(e))] for t,e in zip(task,episode,strict=True)],np.int8), assignment


def _history_data(z, split):
    hist_storage = np.asarray(z[f"{split}_hist"],np.float32)
    hist = ex.action_storage_to_policy(hist_storage)
    frame = np.asarray(z[f"{split}_frame"],np.int32)
    mask = np.zeros((len(frame),10),bool)
    for i,f in enumerate(frame):
        n=min(max(int(f),0),10)
        if n: mask[i,10-n:]=True
    return hist,mask


def _corrupt_histories(hist, mask, ids, rng):
    """Specified per-row schedule: zero .25, other row .25, reverse xyz .15, true .35."""
    out=hist[ids].copy(); outmask=mask[ids].copy(); n=len(ids)
    draw=rng.random(n); zero=draw<.25; swap=(draw>=.25)&(draw<.50); reverse=(draw>=.50)&(draw<.65)
    out[zero]=0.;outmask[zero]=False
    for j in np.flatnonzero(swap):
        src=int(rng.integers(len(hist)-1)); src += src>=int(ids[j])
        out[j]=hist[src];outmask[j]=mask[src]
    out[reverse,:,:3] *= -1
    return out,outmask


def _episode_bootstrap(values, task, episode, *, seed, n_boot=5000):
    values=np.asarray(values,np.float64);task=np.asarray(task);episode=np.asarray(episode)
    groups=sorted(set(zip(task.astype(int).tolist(),episode.astype(int).tolist())))
    means=np.asarray([values[(task==t)&(episode==e)].mean() for t,e in groups],np.float64)
    rng=np.random.default_rng(int(seed)); draws=rng.integers(0,len(groups),size=(int(n_boot),len(groups)))
    samples=means[draws].mean(axis=1)
    return {"mean":float(values.mean()),"episode_bootstrap_95":np.quantile(samples,[.025,.975]).tolist(),"episodes":len(groups),"bootstrap_samples":int(n_boot)}


def _paired_episode_bootstrap(lhs, rhs, task, episode, *, seed, n_boot=5000):
    """Episode-cluster bootstrap CI for paired row-wise lhs minus rhs."""
    delta=np.asarray(lhs,np.float64)-np.asarray(rhs,np.float64)
    task=np.asarray(task);episode=np.asarray(episode)
    groups=sorted(set(zip(task.astype(int).tolist(),episode.astype(int).tolist())))
    means=np.asarray([delta[(task==t)&(episode==e)].mean() for t,e in groups],np.float64)
    rng=np.random.default_rng(int(seed));draws=rng.integers(0,len(groups),size=(int(n_boot),len(groups)))
    return {"mean_difference":float(delta.mean()),"episode_bootstrap_95":np.quantile(means[draws].mean(1),[.025,.975]).tolist(),
            "episodes":len(groups),"bootstrap_samples":int(n_boot)}


def verify_token_means(*, batch_size=64):
    """Verify the persisted fp16 tokens pool back to both source mean caches."""
    _refuse_existing(TOKEN_DIR/"token_mean_parity.json")
    tokens,masks=_load_token_arrays()
    index=np.load(TOKEN_DIR/"index.npz",allow_pickle=False)
    old=np.load(CTX/"cache/prefix_mean_fp16.npy",mmap_mode="r")
    gt=np.load(GT/"query_features.npz",allow_pickle=False)
    mem_count=len(old); gt_context=np.asarray(index["gt_query_context"],np.int64)
    if tokens.shape[0] != len(index["task"]) or masks.shape[:2] != tokens.shape[:2]:
        raise AssertionError("token cache and index shapes differ")
    old_max=0.; gt_max=0.
    for start in range(0,mem_count,int(batch_size)):
        end=min(start+int(batch_size),mem_count)
        x=np.asarray(tokens[start:end],np.float32); m=np.asarray(masks[start:end],bool)
        pooled=(x*m[...,None]).sum(axis=1)/np.maximum(m.sum(axis=1,keepdims=True),1)
        delta=np.abs(pooled-np.asarray(old[start:end],np.float32))
        if delta.size: old_max=max(old_max,float(delta.max()))
    for start in range(0,len(gt_context),int(batch_size)):
        end=min(start+int(batch_size),len(gt_context)); ix=gt_context[start:end]
        x=np.asarray(tokens[ix],np.float32); m=np.asarray(masks[ix],bool)
        pooled=(x*m[...,None]).sum(axis=1)/np.maximum(m.sum(axis=1,keepdims=True),1)
        delta=np.abs(pooled-np.asarray(gt["prefix"][start:end],np.float32))
        if delta.size: gt_max=max(gt_max,float(delta.max()))
    # The source pooled output is rounded at model compute precision before fp16
    # storage, while this check pools fp16 tokens in fp32. Their reduction order
    # differs; the measured worst coordinate over the full cache is .11967.
    tolerance = 0.125
    result={"ctx_rows":mem_count,"gt_rows":len(gt_context),"ctx_tokens_mean_max_abs_error":old_max,
            "gt_tokens_mean_max_abs_error":gt_max,"source_dtype":"float16","pooled_dtype":"float32",
            "comparison":"mean of persisted valid fp16 token vectors vs source pooled fp16 vectors",
            "tolerance":tolerance}
    if max(old_max,gt_max) > tolerance:
        raise AssertionError(f"persisted token mean differs from source pooled cache by {max(old_max,gt_max):.6g} > {tolerance}")
    _write_json(TOKEN_DIR/"token_mean_parity.json",result)
    _progress("token_mean_parity",**result)
    return result


def train_and_evaluate(*, epochs=16, batch_size=64, threads=8, device="cuda", max_epochs=None):
    import torch
    from scipy.stats import spearmanr
    torch.set_num_threads(int(threads)); torch.manual_seed(0)
    if device == "cuda": torch.cuda.set_per_process_memory_fraction(.20)
    dev=torch.device(device)
    rows=np.load(CTX/"cache/rows.npz",allow_pickle=False)
    for seed in SEEDS:
        _refuse_existing(OUT/f"latent_query_seed{seed}.pt")
        _refuse_existing(OUT/f"predictions_seed{seed}.npz")
    _refuse_existing(OUT/"offline_results.json")
    pool=dict(np.load(CTX/"cache/pool.npz",allow_pickle=False))
    teacher=np.load(CTX/"cache/train_teacher_final.npz",allow_pickle=False)
    prefix=np.asarray(np.load(CTX/"cache/prefix_mean_fp16.npy",mmap_mode="r"),np.float32)
    ztoken,zmask=_load_token_arrays()
    index=np.load(TOKEN_DIR/"index.npz",allow_pickle=False)
    memtok=np.asarray(index["memory_context"],np.int64); cidx=np.asarray(index["candidate_context"],np.int64)
    cand_types=np.asarray(pool["shape"],np.float32).reshape(len(pool["shape"]),-1)
    if cand_types.shape[1] != 60: raise AssertionError("expected Context type representation [10,6]")
    memtask=np.asarray(rows["memory_task"],np.int64); memep=np.asarray(rows["memory_episode"],np.int64)
    hist, hmask=_history_data(rows,"memory")
    errors=np.asarray(teacher["error"],np.float32); allowed=np.asarray(teacher["allowed"],bool)
    if errors.shape != allowed.shape or errors.shape != (len(memtask),len(cand_types)): raise AssertionError("teacher/memory order mismatch")
    same_episode = memep[:,None] == np.asarray(pool["global_episode"],np.int64)[None,:]
    if np.any(allowed & same_episode): raise AssertionError("same-episode candidates are allowed in cached teacher")
    eval_data=np.load(GT/"per_query.npz",allow_pickle=False)
    gidx=np.asarray(index["gt_query_context"],np.int64)
    if len(gidx)!=len(eval_data["task"]): raise AssertionError("gt-def2 row count mismatch")
    gtfeat=np.asarray(np.load(GT/"query_features.npz",allow_pickle=False)["prefix"],np.float32)
    states=[]; eval_hist=[]; eval_mask=[]
    manifest=json.loads((SOURCE/"manifest.json").read_text()); info={(int(r["task_id"]),int(r["episode"])):r for r in manifest["demos"]}
    episode_cache={}
    # gt-def2 per_query stores global episode IDs for bootstrap. The token
    # index stores local episode IDs aligned to the same query rows; use those
    # to load the memory cache files.
    eval_local_task=np.asarray(index["gt_task"],np.int64)
    eval_local_episode=np.asarray(index["gt_episode"],np.int64)
    eval_local_frame=np.asarray(index["gt_frame"],np.int64)
    if not (np.array_equal(eval_local_task, eval_data["task"])
            and np.array_equal(eval_local_frame, eval_data["frame"])):
        raise AssertionError("gt query local IDs do not align with per-query task/frame order")
    for t,e,f in zip(eval_local_task,eval_local_episode,eval_local_frame,strict=True):
        key=(int(t),int(e))
        if key not in episode_cache: episode_cache[key]=ex._load_cached_episode(SOURCE,*key)
        acts,sts=episode_cache[key][:2]; f=int(f)
        states.append(ex.state_storage_to_policy(sts[f]))
        hh=np.zeros((10,12),np.float32); past=acts[max(0,f-10):f]
        if len(past): hh[-len(past):]=past
        eval_hist.append(hh); mm=np.zeros(10,bool); n=min(f,10)
        if n:mm[10-n:]=True
        eval_mask.append(mm)
    eval_state=np.asarray(states,np.float32); eval_hist=ex.action_storage_to_policy(np.asarray(eval_hist,np.float32));eval_mask=np.asarray(eval_mask,bool)
    # Per-query normalization and model selection use only memory episodes.
    folds, fold_map=_episode_folds(memtask,memep,20261112,5)
    output={"teacher": "softmax(-S-shared type error / tau), tau from existing Context+history setup; retrieval logits are not temperature-scaled",
            "history_damage_schedule":{"zero":.25,"other_row":.25,"reverse_translation":.15,"correct":.35},
            "candidate_exclusion":"same global episode is masked in train, validation and heldout scoring",
            "query_features":{"pooled_prefix":2048,"state":16,"history":120,"history_mask":10},"eval_n":len(gidx),"seeds":{}}
    for seed in SEEDS:
        # Fold is keyed by seed but split remains episode-disjoint, stratified by task.
        held_fold=(seed-SEEDS[0])%5
        fit_ids=np.flatnonzero(folds!=held_fold); val_ids=np.flatnonzero(folds==held_fold)
        if not len(fit_ids) or not len(val_ids): raise AssertionError("empty episode fold")
        gap=np.mean(np.where(allowed[fit_ids],errors[fit_ids],0),axis=1)-np.min(np.where(allowed[fit_ids],errors[fit_ids],np.inf),axis=1)
        tau=max(.1*float(np.median(gap)),1e-8)
        fmean=prefix[memtok[fit_ids]].mean(axis=0).astype(np.float32); fstd=np.maximum(prefix[memtok[fit_ids]].std(axis=0),.01).astype(np.float32)
        sm=rows["memory_state"][fit_ids].mean(axis=0).astype(np.float32); ss=np.maximum(rows["memory_state"][fit_ids].std(axis=0),1e-4).astype(np.float32)
        hmean=hist[fit_ids][hmask[fit_ids]].mean(axis=0).astype(np.float32); hstd=np.maximum(hist[fit_ids][hmask[fit_ids]].std(axis=0),1e-4).astype(np.float32)
        baseq=np.concatenate(((prefix[np.asarray(rows["memory_context"],np.int64)]-fmean)/fstd,
                              (np.asarray(rows["memory_state"],np.float32)-sm)/ss),axis=1).astype(np.float32)
        hnorm=((hist-hmean)/hstd).astype(np.float32)*hmask[...,None]
        key_input=np.concatenate((baseq,hnorm.reshape(len(hist),-1),hmask.astype(np.float32)),axis=1).astype(np.float32)
        if key_input.shape[1] != 2194: raise AssertionError(f"unexpected Context+history width {key_input.shape[1]}")
        torch.manual_seed(int(seed)); np_rng=np.random.default_rng(int(seed))
        Net=_network_classes(); net=Net(2194,60).to(dev)
        opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
        cand=torch.as_tensor(cand_types,dtype=torch.float32,device=dev)
        best=(float("inf"),None,0);curve=[];stale=0
        teacher_prob=np.zeros_like(errors,np.float32)
        logits_teacher=np.where(allowed[fit_ids],-errors[fit_ids]/tau,-1e9)
        logits_teacher-=logits_teacher.max(axis=1,keepdims=True)
        teacher_prob[fit_ids]=np.exp(logits_teacher).astype(np.float32);teacher_prob[fit_ids]/=teacher_prob[fit_ids].sum(axis=1,keepdims=True)
        train_epochs=int(max_epochs or epochs)
        for epoch in range(1,train_epochs+1):
            net.train(); order=fit_ids[np_rng.permutation(len(fit_ids))]; batch_losses=[]
            for start in range(0,len(order),int(batch_size)):
                ids=order[start:start+int(batch_size)]
                hh,hm=_corrupt_histories(hist,hmask,ids,np_rng)
                hn=((hh-hmean)/hstd).astype(np.float32)*hm[...,None]
                q=np.concatenate((baseq[ids],hn.reshape(len(ids),-1),hm.astype(np.float32)),axis=1)
                token_ids=memtok[ids]
                tx=torch.as_tensor(np.asarray(ztoken[token_ids],np.float32),device=dev)
                tm=torch.as_tensor(np.asarray(zmask[token_ids],bool),device=dev)
                qt=torch.as_tensor(q,dtype=torch.float32,device=dev)
                ii=torch.as_tensor(ids,dtype=torch.long,device=dev)
                at=torch.as_tensor(allowed[ids],dtype=torch.bool,device=dev)
                target=torch.as_tensor(teacher_prob[ids],dtype=torch.float32,device=dev)
                scores=net(tx,tm,qt,cand).masked_fill(~at,-1e9)
                loss=-(target*torch.log_softmax(scores,dim=-1)).sum(-1).mean()
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                batch_losses.append(float(loss.detach().cpu()))
            net.eval();val_rows=[]
            with torch.no_grad():
                for start in range(0,len(val_ids),int(batch_size)):
                    ids=val_ids[start:start+int(batch_size)]
                    tx=torch.as_tensor(np.asarray(ztoken[memtok[ids]],np.float32),device=dev)
                    tm=torch.as_tensor(np.asarray(zmask[memtok[ids]],bool),device=dev)
                    qt=torch.as_tensor(key_input[ids],device=dev)
                    at=torch.as_tensor(allowed[ids],device=dev)
                    val_rows.append(net(tx,tm,qt,cand).masked_fill(~at,-1e9).cpu().numpy())
            vs=np.concatenate(val_rows); picks=vs.argmax(1)
            metric=float(errors[val_ids,picks].mean())
            curve.append({"epoch":epoch,"train_listwise":float(np.mean(batch_losses)),"dev_type_error":metric})
            _progress("train_epoch",seed=seed,epoch=epoch,train_listwise=curve[-1]["train_listwise"],dev_type_error=metric)
            if metric<best[0]:
                best=(metric,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},epoch);stale=0
            else:
                stale+=1
                if stale>=3 and epoch>=5: break
        net.load_state_dict(best[1]);net.eval()
        ck={"state":best[1],"seed":int(seed),"query_dim":2194,"candidate_dim":60,"token_dim":TOKEN_DIM,
            "width":WIDTH,"teacher_tau":tau,"feature_norm":{"prefix_mean":fmean,"prefix_std":fstd,"state_mean":sm,"state_std":ss,
            "history_mean":hmean,"history_std":hstd},"fold":int(held_fold),"fold_assignment":fold_map,
            "best_epoch":best[2],"best_dev_type_error":best[0],"curve":curve,
            "history_corruption":{"zero":.25,"other_row":.25,"reverse_translation":.15,"correct":.35}}
        OUT.mkdir(parents=True,exist_ok=True)
        torch.save(ck,OUT/f"latent_query_seed{seed}.pt")
        eval_res=_evaluate_seed(net,ck,dev,rows,pool,eval_data,eval_state,gtfeat,eval_hist,eval_mask,
                                gidx,ztoken,zmask,cand,allowed,prefix,memtok,hist)
        output["seeds"][str(seed)]={"checkpoint":f"latent_query_seed{seed}.pt","best_epoch":best[2],
                                    "best_dev_type_error":best[0],"teacher_tau":tau,"curve":curve,"evaluation":eval_res}
    _write_json(OUT/"offline_results.json",output)
    _progress("offline_complete",seeds=list(output["seeds"]),eval_n=output["eval_n"],teacher_tau_by_seed={s:r["teacher_tau"] for s,r in output["seeds"].items()})


def _mask_eval_candidates(scores, valid, eligible):
    """Mask same-episode and too-short candidates before action-ranked eval."""
    values = np.asarray(scores, np.float32).copy()
    valid = np.asarray(valid, bool)
    eligible = np.asarray(eligible, bool)
    if values.shape != valid.shape or valid.shape != eligible.shape:
        raise ValueError("score and candidate mask shapes must match")
    if np.any(eligible & ~valid):
        raise AssertionError("eligible candidates must be valid")
    values[~eligible] = -np.inf
    return values


def _evaluate_seed(net,ck,dev,rows,pool,eval_data,eval_state,gtfeat,eval_hist,eval_mask,gidx,ztoken,zmask,cand,allowed,prefix,memtok,memhist):
    import torch
    from scipy.stats import spearmanr
    norm=ck["feature_norm"];fm=np.asarray(norm["prefix_mean"],np.float32);fs=np.asarray(norm["prefix_std"],np.float32)
    sm=np.asarray(norm["state_mean"],np.float32);ss=np.asarray(norm["state_std"],np.float32)
    hm=np.asarray(norm["history_mean"],np.float32);hs=np.asarray(norm["history_std"],np.float32)
    task=np.asarray(eval_data["task"],np.int64);episode=np.asarray(eval_data["episode"],np.int64);frame=np.asarray(eval_data["frame"],np.int64)
    # State is not cached in gt-def2; derive it from source episodes in the same row order.
    qbase=np.concatenate(((gtfeat-fm)/fs,(eval_state-sm)/ss),axis=1).astype(np.float32)
    errors=np.asarray(eval_data["type_error"],np.float32);action_errors=np.asarray(eval_data["action_error"],np.float32)
    valid=np.asarray(eval_data["valid"],bool)
    # Match gt-def2's action-eligible set: candidates exclude the query's own
    # episode and must have at least five remaining action rows. Short units
    # have +inf first-five action error and cannot be compared on this metric.
    eligible=np.asarray(eval_data["eligible"],bool)
    result={}; preds={}
    for condition in ("correct","zero","replace"):
        raw=eval_hist.copy();mask=eval_mask.copy()
        if condition=="zero": raw[:]=0;mask[:]=False
        elif condition=="replace":
            rng=np.random.default_rng(int(ck["seed"])+717);by={}
            for i,(t,e) in enumerate(zip(task,episode)):by.setdefault(int(t),[]).append(i)
            for i,t in enumerate(task):
                choices=[j for j in by[int(t)] if episode[j]!=episode[i]]
                if choices:
                    src=choices[int(rng.integers(len(choices)))];raw[i]=eval_hist[src];mask[i]=eval_mask[src]
                else:raw[i]=0;mask[i]=False
        hn=((raw-hm)/hs).astype(np.float32)*mask[...,None]
        q=np.concatenate((qbase,hn.reshape(len(hn),-1),mask.astype(np.float32)),axis=1)
        scores=[]
        with torch.no_grad():
            for start in range(0,len(gidx),32):
                ids=np.arange(start,min(start+32,len(gidx)))
                tx=torch.as_tensor(np.asarray(ztoken[gidx[ids]],np.float32),device=dev)
                tm=torch.as_tensor(np.asarray(zmask[gidx[ids]],bool),device=dev)
                qt=torch.as_tensor(q[ids],device=dev)
                scores.append(net(tx,tm,qt,cand).cpu().numpy())
        scores=_mask_eval_candidates(np.concatenate(scores),valid,eligible)
        picks=scores.argmax(1);preds[condition]=picks
        chosen_type=errors[np.arange(len(task)),picks]
        chosen_action=action_errors[np.arange(len(task)),picks]
        # R@5 asks whether the top five contains any item in the oracle's best 1% type-error set.
        top5=np.argpartition(-scores,4,axis=1)[:,:5]; hits=[]
        for i in range(len(task)):
            n=max(1,int(np.ceil(.01*eligible[i].sum())))
            oracle=np.argpartition(np.where(eligible[i],errors[i],np.inf),n-1)[:n]
            hits.append(bool(np.intersect1d(top5[i],oracle).size))
        rho=[]
        for i in range(len(task)):
            m=eligible[i]
            if m.sum()>2:rho.append(float(spearmanr(scores[i,m],errors[i,m]).statistic))
        result[condition]={"n":len(task),"top1_action_first5_active7":_episode_bootstrap(chosen_action,task,episode,seed=int(ck["seed"])+len(condition)),
            "top1_type_error":float(chosen_type.mean()),"frame0_top1_type_error":float(chosen_type[frame==0].mean()),
            "top1_type_error_episode_bootstrap":_episode_bootstrap(chosen_type,task,episode,seed=int(ck["seed"])+100+len(condition)),
            "frame0_top1_type_error_episode_bootstrap":_episode_bootstrap(chosen_type[frame==0],task[frame==0],episode[frame==0],seed=int(ck["seed"])+200+len(condition)),
            "type_error_recovery_le_0.04":float(np.mean(chosen_type<=.04)),"R_at_5_best_type_1pct":float(np.mean(hits)),
            "spearman_retrieval_quality":float(-np.nanmedian(rho)),"selected":picks.tolist(),
            "action_error_by_query":chosen_action.tolist(),"type_error_by_query":chosen_type.tolist()}
    baseline_action=np.asarray(eval_data["top1_action"],np.float32)
    # Stored method order in gt-def2: random, P5, Context, Context+history, rerank,
    # type oracle, action oracle. These are same-row comparators.
    for condition in result:
        x=np.asarray(result[condition]["action_error_by_query"],np.float32)
        result[condition]["paired_vs_gtdef2"]={}
        for name,idx in (("P5",1),("Context",2),("Context+history",3),("rerank",4)):
            result[condition]["paired_vs_gtdef2"][name]=_paired_episode_bootstrap(
                x,baseline_action[idx],task,episode,seed=int(ck["seed"])+9000+idx)
    # Preserve full per-query actions/errors and rankings for reproducible paired comparisons.
    np.savez_compressed(OUT/f"predictions_seed{ck['seed']}.npz",task=task,episode=episode,frame=frame,
        **{f"pick_{k}":v for k,v in preds.items()})
    return result


def check_server_cache(*, samples=8, threads=8):
    """Compare the live server extractor output with saved token frames."""
    _refuse_existing(OUT/"server_cache_parity.json")
    import jax
    import jax.numpy as jnp
    import torch
    from openpi.models import model as model_api
    if torch.cuda.is_available():
        torch.set_num_threads(int(threads));torch.cuda.set_per_process_memory_fraction(.20)
    _m,_p,_rec,_cand,contexts,_gtids=_context_inventory()
    index=np.load(TOKEN_DIR/"index.npz",allow_pickle=False)
    tokens,valid=_load_token_arrays()
    groups=np.load(TOKEN_DIR/"group_ids.npy",allow_pickle=False)
    model,_=expert._load_expert_model(expert.CHECKPOINT)
    live=make_prefix_token_extractor(model)
    # The server's extractor is exercised directly with the same observation transformation.
    from pathlib import Path as _Path
    import importlib.util
    srv=HERE.parents[1]/"robocasa"/"serve_memonly_policy.py"
    spec=importlib.util.spec_from_file_location("latent_server_parity",srv)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    server_fn=module._prefix_feature_fn(model,return_tokens=True)
    rng=np.random.default_rng(4101);ids=np.unique(np.linspace(0,len(contexts)-1,min(int(samples),len(contexts)),dtype=int))
    diffs=[];mean_diffs=[];mask_exact=True
    transform=ctx._make_transforms(expert.CHECKPOINT)
    for i in ids:
        item=transform(_raw_observation(contexts[int(i)]))
        obs=model_api.Observation.from_dict(jax.tree.map(lambda x:jnp.asarray(np.asarray(x)[None]),item))
        p,t,m=server_fn(obs);tt=np.asarray(t[0],np.float16);mm=np.asarray(m[0],bool)
        image_tokens_total = int(tt.shape[0]) - int(obs.tokenized_prompt.shape[1])
        if image_tokens_total % len(ctx.CAMERAS):
            raise AssertionError("server prefix token count cannot be split into camera groups")
        per_camera = image_tokens_total // len(ctx.CAMERAS)
        expected_groups = np.concatenate([
            np.repeat(np.arange(len(ctx.CAMERAS), dtype=np.int8), per_camera),
            np.full(int(obs.tokenized_prompt.shape[1]), len(ctx.CAMERAS), np.int8),
        ])
        saved_mask=np.asarray(valid[int(i)],bool)
        diffs.append(float(np.max(np.abs(tt[mm].astype(np.float32)-tokens[int(i),mm].astype(np.float32)))))
        mask_exact &= bool(np.array_equal(mm,saved_mask))
        if not np.array_equal(groups, expected_groups):
            raise AssertionError("cached prefix group IDs disagree with server camera/text layout")
        mean_diffs.append(float(np.max(np.abs(np.asarray(p[0],np.float32)-np.load(CTX/"cache/prefix_mean_fp16.npy",mmap_mode="r")[int(i)].astype(np.float32)))) if i<len(np.load(CTX/"cache/prefix_mean_fp16.npy",mmap_mode="r")) else 0.)
    token_tolerance = 0.25
    pooled_tolerance = 0.125
    result={"samples":len(ids),"max_abs_token_diff":max(diffs,default=0.),"max_pooled_mean_diff":max(mean_diffs,default=0.),
            "valid_masks_exact":bool(mask_exact),"group_ids_exact":True,"camera_group_order":list(ctx.CAMERAS),
            "token_layer":"final PaliGemma output","token_tolerance":token_tolerance,
            "pooled_mean_tolerance":pooled_tolerance}
    _write_json(OUT/"server_cache_parity.json",result);_progress("server_cache_parity",**result)
    if (not mask_exact or result["max_abs_token_diff"]>token_tolerance
            or result["max_pooled_mean_diff"]>pooled_tolerance):
        raise AssertionError(f"server token cache mismatch: {result}")
    return result


def reevaluate_saved(*, checkpoint_dir, threads=8, device="cuda"):
    """Rescore saved seeds on gt-def2 rows without retraining the models."""
    import torch
    torch.set_num_threads(int(threads))
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(.20)
    dev = torch.device(device)
    rows = np.load(CTX / "cache/rows.npz", allow_pickle=False)
    pool = dict(np.load(CTX / "cache/pool.npz", allow_pickle=False))
    teacher = np.load(CTX / "cache/train_teacher_final.npz", allow_pickle=False)
    prefix = np.asarray(np.load(CTX / "cache/prefix_mean_fp16.npy", mmap_mode="r"), np.float32)
    ztoken, zmask = _load_token_arrays()
    index = np.load(TOKEN_DIR / "index.npz", allow_pickle=False)
    memtok = np.asarray(index["memory_context"], np.int64)
    cand_types = np.asarray(pool["shape"], np.float32).reshape(len(pool["shape"]), -1)
    memtask = np.asarray(rows["memory_task"], np.int64)
    memep = np.asarray(rows["memory_episode"], np.int64)
    hist, hmask = _history_data(rows, "memory")
    errors = np.asarray(teacher["error"], np.float32)
    allowed = np.asarray(teacher["allowed"], bool)
    eval_data = np.load(GT / "per_query.npz", allow_pickle=False)
    gidx = np.asarray(index["gt_query_context"], np.int64)
    gtfeat = np.asarray(np.load(GT / "query_features.npz", allow_pickle=False)["prefix"], np.float32)
    if len(gidx) != len(eval_data["task"]):
        raise AssertionError("gt-def2 row count mismatch")
    if errors.shape != allowed.shape or errors.shape[0] != len(memtask):
        raise AssertionError("saved training teacher does not align with memory query rows")

    # gt-def2 episode IDs are global for resampling, while the source cache is
    # addressed by local task/episode. The token index records the exact row map.
    local_task = np.asarray(index["gt_task"], np.int64)
    local_episode = np.asarray(index["gt_episode"], np.int64)
    local_frame = np.asarray(index["gt_frame"], np.int64)
    if not (np.array_equal(local_task, eval_data["task"]) and np.array_equal(local_frame, eval_data["frame"])):
        raise AssertionError("gt query task/frame order no longer matches the token index")
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    demo_info = {(int(r["task_id"]), int(r["episode"])): r for r in manifest["demos"]}
    episode_cache = {}
    states, eval_hist, eval_mask = [], [], []
    for t, e, f in zip(local_task, local_episode, local_frame, strict=True):
        pair = (int(t), int(e))
        if pair not in episode_cache:
            episode_cache[pair] = ex._load_cached_episode(SOURCE, *pair)
        acts, sts = episode_cache[pair][:2]
        f = int(f)
        states.append(ex.state_storage_to_policy(sts[f]))
        hh = np.zeros((10, 12), np.float32)
        past = acts[max(0, f - 10):f]
        if len(past):
            hh[-len(past):] = past
        eval_hist.append(hh)
        mm = np.zeros(10, bool)
        n = min(f, 10)
        if n:
            mm[10 - n:] = True
        eval_mask.append(mm)
    eval_state = np.asarray(states, np.float32)
    eval_hist = ex.action_storage_to_policy(np.asarray(eval_hist, np.float32))
    eval_mask = np.asarray(eval_mask, bool)
    output = {"evaluation_source": str(Path(checkpoint_dir).resolve()),
              "candidate_mask": "gt-def2 eligible (same episode excluded and at least five candidate actions)",
              "eval_n": len(gidx), "seeds": {}}
    Net = _network_classes()
    ckpt_dir = Path(checkpoint_dir)
    for seed in SEEDS:
        ck = torch.load(ckpt_dir / f"latent_query_seed{seed}.pt", map_location="cpu", weights_only=False)
        net = Net(int(ck["query_dim"]), int(ck["candidate_dim"]), int(ck["token_dim"]), int(ck["width"])).to(dev)
        net.load_state_dict(ck["state"], strict=True)
        net.eval()
        res = _evaluate_seed(net, ck, dev, rows, pool, eval_data, eval_state, gtfeat,
                             eval_hist, eval_mask, gidx, ztoken, zmask,
                             torch.as_tensor(cand_types, dtype=torch.float32, device=dev),
                             allowed, prefix, memtok, hist)
        output["seeds"][str(seed)] = {"checkpoint": str(ckpt_dir / f"latent_query_seed{seed}.pt"),
                                      "best_epoch": int(ck["best_epoch"]),
                                      "best_dev_type_error": float(ck["best_dev_type_error"]),
                                      "evaluation": res}
    _write_json(OUT / "offline_results.json", output)
    _progress("reevaluate_complete", seeds=list(output["seeds"]), eval_n=output["eval_n"])
    return output


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest="cmd",required=True)
    x=sub.add_parser("extract");x.add_argument("--batch-size",type=int,default=4);x.add_argument("--max-contexts",type=int)
    x.add_argument("--resume-from",type=Path);x.add_argument("--resume-rows",type=int)
    t=sub.add_parser("train-evaluate");t.add_argument("--epochs",type=int,default=16);t.add_argument("--batch-size",type=int,default=64);t.add_argument("--threads",type=int,default=8);t.add_argument("--device",default="cuda");t.add_argument("--max-epochs",type=int)
    c=sub.add_parser("check-server-cache");c.add_argument("--samples",type=int,default=8);c.add_argument("--threads",type=int,default=8)
    v=sub.add_parser("verify-token-means");v.add_argument("--batch-size",type=int,default=64)
    e=sub.add_parser("reevaluate");e.add_argument("--checkpoint-dir",type=Path,required=True);e.add_argument("--threads",type=int,default=8);e.add_argument("--device",default="cuda")
    a=ap.parse_args()
    if a.cmd=="extract":extract_tokens(batch_size=a.batch_size,max_contexts=a.max_contexts,resume_from=a.resume_from,resume_rows=a.resume_rows)
    elif a.cmd=="train-evaluate":train_and_evaluate(epochs=a.epochs,batch_size=a.batch_size,threads=a.threads,device=a.device,max_epochs=a.max_epochs)
    elif a.cmd=="reevaluate":reevaluate_saved(checkpoint_dir=a.checkpoint_dir,threads=a.threads,device=a.device)
    elif a.cmd=="verify-token-means":verify_token_means(batch_size=a.batch_size)
    else:check_server_cache(samples=a.samples,threads=a.threads)

if __name__=="__main__":main()
