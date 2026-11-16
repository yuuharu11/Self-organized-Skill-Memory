#!/usr/bin/env python3
"""LA1: compare decoding latent actions with retrieving stored motion units.

The implementation reuses KEY2's fixed row/data contract and rank metrics but
keeps LA-specific models and checkpoints in a separate output tree.  Learned
and unseen evaluation targets are never part of fitting or Dev selection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
KEY2_SNAPSHOT = Path('/work/.skillnet-state/snapshots/key2f-20261114T181328Z/skill_moe/skillnet/examples/memory_pilot/unseen_libero')
if str(KEY2_SNAPSHOT) not in sys.path:
    sys.path.insert(0, str(KEY2_SNAPSHOT))
import key_shared_latent as key2  # noqa: E402

SEED = 4101
BOOTSTRAP_REPS = 5000
TEACHER_TAU = 0.01
VQ_CODES = (64, 256)
LEARNING_RATES = (1e-4, 3e-4)
EPOCHS = (4, 8, 16)


def pooled_tokens(tokens: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Pool valid tokens robustly, preserving a fixed-size observation vector."""
    tokens = np.asarray(tokens)
    groups = np.asarray(groups)
    if tokens.ndim != 3 or groups.shape != tokens.shape[:2]:
        raise ValueError('tokens/groups must be [N,T,D]/[N,T]')
    mask = groups >= 0
    out = np.zeros((len(tokens), tokens.shape[-1]), np.float32)
    for lo in range(0, len(tokens), 32):
        hi = min(len(tokens), lo + 32)
        block = np.where(mask[lo:hi, :, None], tokens[lo:hi], 0)
        out[lo:hi] = block.sum(axis=1, dtype=np.float32) / np.maximum(mask[lo:hi].sum(1, keepdims=True), 1)
    return out


def vqvae_loss(reconstruction: torch.Tensor, target: torch.Tensor,
               quantized: torch.Tensor, encoded: torch.Tensor, beta: float = 0.25) -> torch.Tensor:
    """Standard masked-free fixed-horizon VQ-VAE loss (all values are valid)."""
    if reconstruction.shape != target.shape or quantized.shape != encoded.shape:
        raise ValueError('VQ-VAE tensors must have matching shapes')
    return F.mse_loss(reconstruction, target) + F.mse_loss(quantized.detach(), encoded) + beta * F.mse_loss(quantized, encoded.detach())


class ActionVQVAE(nn.Module):
    """VQ-VAE over 10-step, six-axis arm chunks; gripper is scored from truth."""
    def __init__(self, codes: int, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(60, 256), nn.GELU(), nn.Linear(256, latent_dim))
        self.codebook = nn.Embedding(codes, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1 / codes, 1 / codes)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.GELU(), nn.Linear(256, 60))

    def encode(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(actions.reshape(len(actions), 60))
        # Squared Euclidean nearest code; computation is stable in float32.
        d = (z.square().sum(1, keepdim=True) + self.codebook.weight.square().sum(1)[None]
             - 2 * z @ self.codebook.weight.T)
        ids = d.argmin(1)
        q = self.codebook(ids)
        return ids, z, q

    def forward(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ids, z, q = self.encode(actions)
        rec = self.decoder(q).reshape(-1, 10, 6)
        return ids, z, q, rec

    def decode_code(self, ids: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.codebook(ids)).reshape(-1, 10, 6)


class QueryEncoder(nn.Module):
    """Shared KEY2-style observation query network for LA retrieval/decoding."""
    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.GELU(), nn.LayerNorm(512),
                                 nn.Linear(512, 256), nn.GELU(), nn.Linear(256, latent_dim))
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(10.0))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class UnitKeyEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.GELU(), nn.LayerNorm(512),
                                 nn.Linear(512, 256), nn.GELU(), nn.Linear(256, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ActionDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 256), nn.GELU(), nn.Linear(256, 60))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).reshape(-1, 10, 6)


def teacher_distribution(errors: torch.Tensor, allowed: torch.Tensor, tau: float = TEACHER_TAU) -> torch.Tensor:
    if errors.shape != allowed.shape:
        raise ValueError('teacher errors and candidate mask must agree')
    logits = (-errors.float() / float(tau)).masked_fill(~allowed.bool(), -1e9)
    return torch.softmax(logits, dim=-1)


def listwise_loss(scores: torch.Tensor, errors: torch.Tensor, allowed: torch.Tensor,
                  tau: float = TEACHER_TAU) -> torch.Tensor:
    if scores.shape != errors.shape or scores.shape != allowed.shape:
        raise ValueError('score, error, and allowed arrays must match')
    target = teacher_distribution(errors, allowed, tau)
    logp = torch.log_softmax(scores.float().masked_fill(~allowed.bool(), -1e9), dim=-1)
    valid = allowed.any(dim=-1)
    return -(target * logp).sum(-1)[valid].mean()


def exclude_self_episode(query_task: Any, query_episode: Any,
                         candidate_task: Any, candidate_episode: Any) -> np.ndarray:
    """KEY2-compatible self-episode exclusion; no action label is consulted."""
    return ~((np.asarray(query_task)[:, None] == np.asarray(candidate_task)[None]) &
             (np.asarray(query_episode)[:, None] == np.asarray(candidate_episode)[None]))


def topk_blend(scores: np.ndarray, actions: np.ndarray, k: int,
               temperature: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Softmax blend of top-k retrieved *stored actions*, never latent values."""
    scores = np.asarray(scores, np.float32)
    actions = np.asarray(actions, np.float32)
    k = max(1, min(int(k), scores.shape[1]))
    order = np.argsort(-scores, axis=1, kind='stable')[:, :k]
    local = np.take_along_axis(scores, order, axis=1)
    valid = np.isfinite(local)
    local = np.where(valid, local, -np.inf)
    row_max = np.max(local, axis=1, keepdims=True)
    weights = np.exp((local - row_max) / max(float(temperature), 1e-6))
    weights = np.where(valid, weights, 0.)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return np.einsum('qk,qkta->qta', weights, actions[order]), order


def ranking_metrics(scores: np.ndarray, errors: np.ndarray, tau: float = 0.01,
                    score_temperature: float = 1.0) -> dict[str, Any]:
    """KEY2-compatible graded rank metrics, restricted to allowed candidates."""
    from scipy.stats import spearmanr
    scores = np.asarray(scores, np.float64); errors = np.asarray(errors, np.float64)
    if scores.shape != errors.shape or scores.ndim != 2 or tau <= 0 or score_temperature <= 0:
        raise ValueError('scores/errors must be [Q,N] and temperatures positive')
    allowed = np.isfinite(scores)
    if np.any(~allowed.any(1)):
        raise ValueError('each query needs at least one allowed candidate')
    n, m = scores.shape; row = np.arange(n)
    masked_scores = np.where(allowed, scores, -np.inf)
    masked_errors = np.where(allowed, errors, np.inf)
    pred_order = np.argsort(-masked_scores, axis=1, kind='stable')
    true_order = np.argsort(masked_errors, axis=1, kind='stable')
    pred_prob = key2._softmax_np(masked_scores / float(score_temperature), axis=1)
    teacher_prob = key2._softmax_np(-masked_errors / float(tau), axis=1)
    cross_entropy = float(np.mean(-np.sum(teacher_prob * np.log(np.maximum(pred_prob, 1e-300)), axis=1)))
    kl = float(np.mean(np.sum(teacher_prob * (np.log(np.maximum(teacher_prob, 1e-300)) -
                                               np.log(np.maximum(pred_prob, 1e-300))), axis=1)))
    correlations = []
    for i in range(n):
        valid = allowed[i]
        if valid.sum() > 1 and np.ptp(scores[i, valid]) > 0 and np.ptp(errors[i, valid]) > 0:
            correlations.append(float(spearmanr(scores[i, valid], -errors[i, valid]).statistic))
    recalls = {}
    for k0 in (1, 5, 20):
        per_query = []
        for i in range(n):
            k = min(k0, int(allowed[i].sum()))
            per_query.append(len(set(true_order[i, :k]) & set(pred_order[i, :k])) / k)
        recalls[str(k0)] = float(np.mean(per_query))
    k20 = min(20, m); discount = np.log2(np.arange(2, k20 + 2))[None, :]
    relevance = np.where(allowed, np.exp(-errors / float(tau)), 0.)
    dcg = np.sum(np.take_along_axis(relevance, pred_order[:, :k20], axis=1) / discount, axis=1)
    ideal = np.sum(np.take_along_axis(relevance, true_order[:, :k20], axis=1) / discount, axis=1)
    ndcg = float(np.mean(dcg / np.maximum(ideal, 1e-300)))
    best = true_order[:, 0]; best_rank = np.argmax(pred_order == best[:, None], axis=1) + 1
    top1 = masked_errors[row, pred_order[:, 0]]
    top20 = []; random_top1 = []; rng = np.random.default_rng(0)
    for i in range(n):
        count = min(20, int(allowed[i].sum()))
        top20.append(float(np.mean(masked_errors[i, pred_order[i, :count]])))
        random_top1.append(float(errors[i, rng.choice(np.flatnonzero(allowed[i]))]))
    oracle_top1 = masked_errors[row, best]; random_mean = float(np.mean(random_top1))
    oracle_mean = float(np.mean(oracle_top1))
    return {'spearman_score_neg_error': float(np.mean(correlations)) if correlations else 0.,
        'H_pq_cross_entropy': cross_entropy, 'kl_teacher_to_prediction': kl,
        'recall_at_k': recalls, 'ndcg_at_20': ndcg, 'mrr': float(np.mean(1. / best_rank)),
        'top1_arm6_mse': float(np.mean(top1)), 'top20_arm6_mse': float(np.mean(top20)),
        'oracle_top1_arm6_mse': oracle_mean, 'random_top1_arm6_mse': random_mean,
        'key2_recovery': float((random_mean - np.mean(top1)) / max(random_mean - oracle_mean, 1e-12)),
        'ranking_reference_tau': float(tau), 'score_temperature': float(score_temperature)}


def episode_bootstrap(values: np.ndarray, task: np.ndarray, episode: np.ndarray,
                      reps: int = BOOTSTRAP_REPS, seed: int = SEED) -> dict[str, float]:
    return key2.episode_bootstrap_ci(np.asarray(values), np.asarray(task), np.asarray(episode), reps=reps, seed=seed)


def action_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """MSE over arm6, with oracle gripper copied from the target (KEY2 metric)."""
    return np.square(np.asarray(pred)[..., :6] - np.asarray(target)[..., :6]).sum(axis=(1, 2)) / (10 * 7)


def soft_blend_from_scores(scores: np.ndarray, candidate_actions: np.ndarray,
                           k: int, temperature: float) -> np.ndarray:
    return topk_blend(scores, candidate_actions, k, temperature)[0]


def select_blend_temperatures(dev_scores: np.ndarray, candidate_actions: np.ndarray,
                              dev_targets: np.ndarray,
                              temperatures=(0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0)) -> dict[str, dict[str, float]]:
    """Choose a separate top-k softmax temperature using Dev action error only."""
    selected = {}
    for k in (1, 5, 20):
        rows=[]
        for temperature in ((1.0,) if k == 1 else tuple(temperatures)):
            pred,_=topk_blend(dev_scores,candidate_actions,k,temperature)
            rows.append({'k':int(k),'temperature':float(temperature),
                         'dev_mse_gripper_truth':float(action_error(pred,dev_targets).mean())})
        selected[str(k)]=min(rows,key=lambda x:(x['dev_mse_gripper_truth'],x['temperature']))
    return selected


def make_observation_matrix(split: dict[str, np.ndarray], *, vjepa: bool = False,
                            zero_history: bool = False) -> np.ndarray:
    """Build fixed, label-free query input from frozen pooled features/context."""
    vlm = pooled_tokens(split['tokens'], split['groups'])
    items = [vlm]
    if vjepa:
        items.append(pooled_tokens(split['vjepa_tokens'], split['vjepa_groups']))
    history = np.asarray(split['history'], np.float32).copy()
    mask = np.asarray(split['history_mask'], np.float32)
    if zero_history:
        history[:] = 0; mask[:] = 0
    items.extend([history.reshape(len(history), -1), mask, np.asarray(split['state'], np.float32)])
    return np.concatenate(items, axis=1).astype(np.float32)


def make_unit_input(bank: dict[str, np.ndarray], bank_input: dict[str, np.ndarray],
                    *, use_action: bool, use_vjepa: bool = False,
                    end_observation: np.ndarray | None = None) -> np.ndarray:
    """Unit key input: action + exact start/end observations; IDM excludes action."""
    starts = pooled_tokens(bank_input['tokens'], bank_input['groups'])
    ends = starts if end_observation is None else np.asarray(end_observation, np.float32)
    fields = [starts, ends]
    if use_vjepa:
        vj = pooled_tokens(bank_input['vjepa_tokens'], bank_input['vjepa_groups'])
        fields.append(vj)
    if use_action:
        fields.append(np.asarray(bank['value_actions10'], np.float32)[..., :6].reshape(len(starts), -1))
    return np.concatenate(fields, axis=1).astype(np.float32)


def extract_unit_end_prefixes(keys: np.ndarray, output: Path, *,
                              run_root: Path, primary_cache: Path,
                              secondary_cache: Path, extra_caches: tuple[Path, ...] = (),
                              checkpoint: Path | None = None,
                              batch_size: int = 4) -> dict[str, Any]:
    """Extract exact unit-end prefix rows that are absent from read-only caches.

    This mirrors KEY2's π0.5 prefix transform path and never fills gaps with a
    neighboring frame. Existing rows are used for an exact parity check.
    """
    import importlib.util
    import io
    from collections import OrderedDict
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.05')
    import pyarrow.parquet as pq
    from PIL import Image
    import jax
    from openpi.transforms import Normalize, ResizeImages, TokenizePrompt, PadStatesAndActions
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.models.model import ModelType
    from openpi.shared.normalize import load as load_norm
    from openpi.shared.nnx_utils import module_jit

    output=Path(output); run_root=Path(run_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite unit-end feature output {output}')
    output.mkdir(parents=True,exist_ok=True)
    requested=sorted({tuple(map(int,k)) for k in np.asarray(keys)})
    def read_cache(path: Path):
        path=Path(path)
        if (path/'manifest.json').exists():
            m=json.loads((path/'manifest.json').read_text())
            return [tuple(map(int,k)) for k in m['keys']], np.load(path/'tokens.npy',mmap_mode='r'), np.load(path/'groups.npy',mmap_mode='r')
        row={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (path/'rows').glob('*.npy')}
        k=list(zip(row['task'].tolist(),row['episode'].tolist(),row['frame'].tolist()))
        return k,np.load(path/'tokens.npy',mmap_mode='r'),np.load(path/'groups.npy',mmap_mode='r')
    cache_data=[read_cache(path) for path in (primary_cache,secondary_cache,*extra_caches)]
    maps=[({tuple(map(int,k)):i for i,k in enumerate(cache[0])},cache[1],cache[2]) for cache in cache_data]
    known={k for mapping,_,_ in maps for k in mapping}
    missing=sorted(set(requested)-known)
    overlap=sorted(set(requested)&known)
    verify=overlap[:min(8,len(overlap))]
    required=sorted(set(missing)|set(verify))
    data_manifest=json.loads((run_root/'data_manifest.json').read_text())
    source=Path(data_manifest['source']); ckpt=Path(checkpoint or data_manifest['checkpoint'])
    records=data_manifest['episodes'];record_map={(int(r['task']),int(r['episode'])):r for r in records}
    tasks={int(k):v for k,v in data_manifest['tasks'].items()}
    norm_files=list((ckpt/'assets').rglob('norm_stats.json'))
    if len(norm_files)!=1: raise ValueError(f'expected one norm_stats.json, got {norm_files}')
    transforms=[LiberoInputs(ModelType.PI05),Normalize(load_norm(norm_files[0].parent),use_quantiles=True),
        ResizeImages(224,224),TokenizePrompt(PaligemmaTokenizer(200),discrete_state_input=False),PadStatesAndActions(32)]
    pi05_dir=HERE.parent/'pi05_memory';run_py=pi05_dir/'run.py'
    spec=importlib.util.spec_from_file_location('la_pi05_run',run_py)
    if spec is None or spec.loader is None: raise ImportError(run_py)
    run_mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=run_mod;spec.loader.exec_module(run_mod)
    model,_=run_mod.load_policy(ckpt); encode=module_jit(model.prefix_features)
    parquet_cache: OrderedDict[Path,Any]=OrderedDict()
    def observe(key):
        task,episode,frame=key; rec=record_map[(task,episode)]; p=Path(rec['path'])
        if p not in parquet_cache:
            if len(parquet_cache)>=2: parquet_cache.popitem(last=False)
            parquet_cache[p]=pq.read_table(p,columns=['image','wrist_image','state','actions'])
        table=parquet_cache[p]; images={}
        for name in ('image','wrist_image'):
            obj=table[name][frame].as_py()
            with Image.open(io.BytesIO(obj['bytes'])) as im: images[name]=np.asarray(im.convert('RGB'))
        item={'observation/image':images['image'],'observation/wrist_image':images['wrist_image'],
            'observation/state':np.asarray(table['state'][frame].as_py(),np.float32),
            'actions':np.asarray(table['actions'].slice(frame,10).to_pylist(),np.float32),'prompt':tasks[task]}
        for tr in transforms: item=tr(item)
        return {'image_base':np.asarray(item['image']['base_0_rgb']),'image_wrist':np.asarray(item['image']['left_wrist_0_rgb']),
            'state':np.asarray(item['state']),'prompt':np.asarray(item['tokenized_prompt']),
            'prompt_mask':np.asarray(item['tokenized_prompt_mask'])}
    def encode_batch(items):
        packed={k:np.stack([x[k] for x in items]) for k in items[0]}
        obs={'image_base':packed['image_base'],'image_wrist':packed['image_wrist'],'state':packed['state'],
             'prompt':packed['prompt'],'prompt_mask':packed['prompt_mask']}
        hidden,groups=encode(run_mod.obs_batch(obs,np.arange(len(items))))
        return np.asarray(hidden),np.asarray(groups)
    import seg_context_extract as extract
    extract.extract_prefix_memmap(required,observe,encode_batch,output,batch_size=batch_size,
        token_shape=(532,2048),valid_only=True)
    exkeys=[tuple(map(int,k)) for k in json.loads((output/'manifest.json').read_text())['keys']]
    exmap={k:i for i,k in enumerate(exkeys)}
    et=np.load(output/'tokens.npy',mmap_mode='r'); eg=np.load(output/'groups.npy',mmap_mode='r')
    errors=[]
    for key in verify:
        i=exmap[key]
        for cache,(mapping,_,_) in zip(cache_data,maps,strict=True):
            if key not in mapping: continue
            j=mapping[key];valid=np.asarray(cache[2][j])>=0;n=int(valid.sum())
            if not np.array_equal(eg[i,:n],np.asarray(cache[2][j])[valid]): raise AssertionError(f'group parity failed for {key}')
            delta=np.abs(np.asarray(et[i,:n],np.float32)-np.asarray(cache[1][j,valid],np.float32))
            errors.append(float(delta.max(initial=0)))
    if verify and (not errors or max(errors)>0): raise AssertionError(f'exact prefix parity failed: {errors}')
    report={'requested_keys':len(requested),'missing_before':len(missing),'reextracted_for_parity':len(verify),
        'max_abs_parity_error':max(errors) if errors else None,'extracted_keys':len(required),
        'feature_shape':list(et.shape),'groups_shape':list(eg.shape),'complete':True}
    (output/'extract_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def exact_pool_with_extra(keys: np.ndarray, store: Any, extra_dir: Path) -> np.ndarray:
    """Average exact cached prefix tokens, with an exact extracted-key supplement."""
    extra_dir=Path(extra_dir)
    exkeys=[tuple(map(int,k)) for k in json.loads((extra_dir/'manifest.json').read_text())['keys']]
    exmap={k:i for i,k in enumerate(exkeys)}
    et=np.load(extra_dir/'tokens.npy',mmap_mode='r');eg=np.load(extra_dir/'groups.npy',mmap_mode='r')
    result=np.empty((len(keys),2048),np.float32)
    caches=[x for x in store.caches if x[0]=='libero40']
    for i,key0 in enumerate(keys):
        key=tuple(map(int,key0)); found=None
        for _,mapping,tokens,groups in caches:
            if key in mapping:
                j=mapping[key];valid=np.asarray(groups[j])>=0
                tok=np.asarray(tokens[j])[valid]
                if found is not None and not np.array_equal(found,tok): raise ValueError(f'exact duplicate cache mismatch: {key}')
                found=tok
        if found is None:
            if key not in exmap: raise KeyError(f'exact endpoint missing: {key}')
            j=exmap[key];valid=np.asarray(eg[j])>=0;found=np.asarray(et[j])[valid]
        if not len(found): raise ValueError(f'no valid token at endpoint: {key}')
        result[i]=found.astype(np.float32).mean(0)
    return result


def train_vqvae(actions: np.ndarray, codes: int, *, epochs: int,
                learning_rate: float, device: str, seed: int = SEED) -> tuple[ActionVQVAE, dict[str, Any]]:
    torch.manual_seed(seed); np.random.seed(seed)
    model = ActionVQVAE(codes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    x = torch.as_tensor(np.asarray(actions, np.float32)[..., :6], device=device)
    history = []
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        order = rng.permutation(len(x)); losses=[]
        model.train()
        for ix in np.array_split(order, max(1, int(np.ceil(len(order) / 128)))):
            xb = x[ix]
            ids, z, q, rec = model(xb)
            loss = vqvae_loss(rec, xb, q, z)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    model.eval()
    with torch.no_grad():
        ids, _, _, _ = model(x)
    used = torch.unique(ids).numel()
    return model, {'loss': history, 'requested_codes': codes, 'used_codes': int(used)}


def choose_dev(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Select solely on Dev: minimum arm6 top-1 action error, deterministic tie-break."""
    if not results:
        raise ValueError('no Dev results available')
    return min(results, key=lambda x: (x['dev_top1_arm6_mse'], x['config_id']))


def _seed_everything(seed: int) -> None:
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _as_t(x, device):
    return torch.as_tensor(np.asarray(x, np.float32), device=device)


def _predict_latents(qnet: QueryEncoder, x: np.ndarray, device: str,
                     batch_size: int = 128) -> np.ndarray:
    qnet.eval(); out=[]
    with torch.no_grad():
        for lo in range(0, len(x), batch_size): out.append(qnet(_as_t(x[lo:lo+batch_size], device)).cpu().numpy())
    return np.concatenate(out)


def _encode_keys(knet: UnitKeyEncoder, x: np.ndarray, device: str,
                 batch_size: int = 256) -> np.ndarray:
    knet.eval(); out=[]
    with torch.no_grad():
        for lo in range(0, len(x), batch_size): out.append(knet(_as_t(x[lo:lo+batch_size], device)).cpu().numpy())
    return np.concatenate(out)


def _fit_code_head(qnet: QueryEncoder, qx: np.ndarray, labels: np.ndarray,
                   *, epochs: int, learning_rate: float, device: str, seed: int) -> nn.Linear:
    # Attach an explicit classifier to the query net. Kept as a helper so the
    # classifier cannot accidentally see any evaluation actions.
    head = nn.Linear(128, int(np.max(labels)) + 1).to(device)
    qnet.to(device); opt = torch.optim.AdamW(list(qnet.parameters()) + list(head.parameters()), lr=learning_rate)
    x = _as_t(qx, device); y = torch.as_tensor(labels, device=device, dtype=torch.long)
    rng = np.random.default_rng(seed); ids = np.arange(len(x))
    for _ in range(epochs):
        rng.shuffle(ids)
        for ix in np.array_split(ids, max(1, int(np.ceil(len(ids) / 64)))):
            ti = torch.as_tensor(ix, device=device, dtype=torch.long)
            loss = F.cross_entropy(head(qnet(x[ti])), y[ti])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return head


def _score_and_decode(qnet, knet, decoder, qx, kx, device, code_head=None, vq=None,
                      candidate_codes=None):
    q = _predict_latents(qnet, qx, device)
    k = _encode_keys(knet, kx, device)
    scores = q @ k.T
    if code_head is not None:
        with torch.no_grad(): logits = code_head(_as_t(q, device)).cpu().numpy()
        pred_codes = logits.argmax(1)
        scores[:, np.asarray(candidate_codes) != pred_codes[:, None]] = -np.inf
        return scores, pred_codes
    return scores, None


def _bootstrap_summary(errors: np.ndarray, split: dict[str, np.ndarray]) -> dict[str, Any]:
    a = np.asarray(errors, np.float32)
    out = {'mean': float(a.mean()), 'median': float(np.median(a)),
           'episode_bootstrap_95': episode_bootstrap(a, split['task'], split['episode'])}
    hard = np.asarray(split.get('hard', np.zeros(len(a), bool)), bool)
    out['hard'] = ({'n': int(hard.sum()), 'mean': float(a[hard].mean()),
                    'median': float(np.median(a[hard])),
                    'episode_bootstrap_95': episode_bootstrap(a[hard], np.asarray(split['task'])[hard],
                                                               np.asarray(split['episode'])[hard])}
                   if hard.any() else {'n': 0, 'mean': None, 'median': None})
    return out


def _rank_by_slice(scores: np.ndarray, errors: np.ndarray, split: dict[str, np.ndarray],
                   score_temperature: float = 1.0) -> dict[str, Any]:
    hard = np.asarray(split.get('hard', np.zeros(len(scores), bool)), bool)
    return {'all': ranking_metrics(scores, errors, score_temperature=score_temperature),
            'hard': ranking_metrics(scores[hard], errors[hard], score_temperature=score_temperature) if hard.any() else None}


def select_rank_temperature(dev_scores: np.ndarray, dev_errors: np.ndarray,
                            temperatures=(0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0)) -> dict[str, float]:
    """Select score softmax temperature on Dev by the fixed-τ cross entropy."""
    scores=np.asarray(dev_scores,np.float64);errors=np.asarray(dev_errors,np.float64)
    allowed=np.isfinite(scores)
    if scores.shape!=errors.shape or np.any(~allowed.any(1)):
        raise ValueError('Dev scores/errors are misaligned or have no allowed candidates')
    teacher=key2._softmax_np(np.where(allowed,-errors/TEACHER_TAU,-np.inf),axis=1)
    rows=[]
    for temperature in temperatures:
        pred=key2._softmax_np(np.where(allowed,scores/float(temperature),-np.inf),axis=1)
        ce=float(np.mean(-np.sum(teacher*np.log(np.maximum(pred,1e-300)),axis=1)))
        rows.append({'temperature':float(temperature),'dev_cross_entropy':ce})
    return min(rows,key=lambda x:(x['dev_cross_entropy'],x['temperature']))


def run_experiment(output: Path, *, run_root: Path, missing_root: Path,
                   seen_stride5: Path, unseen_stride5: Path, vjepa_eval_cache: Path,
                   device: str = 'cuda:0', memory_fraction: float = .18,
                   epochs: int = 4, learning_rate: float = 3e-4,
                   codes: tuple[int, ...] = VQ_CODES, smoke: bool = False,
                   unit_end_features: Path | None = None) -> dict[str, Any]:
    """Run the fixed-split LA comparison. Existing output is never overwritten."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite non-empty output {output}')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    if torch.cuda.is_available() and str(device).startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(float(memory_fraction), device=torch.device(device))
    _seed_everything(SEED)
    bank, bank_input, fit_bank, fit_bank_input, train, learned, unseen, dev = key2._build_key2_queries(
        run_root, missing_root, seen_stride5, unseen_stride5, vjepa_eval_cache)
    if len(bank['length']) != 4516 or len(learned['key']) != 621 or len(unseen['key']) != 619:
        raise AssertionError('fixed KEY2 rows / unit pool changed')
    if smoke:
        nsmoke=min(48,len(train['key']))
        source_map={tuple(map(int,k)):i for i,k in enumerate(bank_input['key'])}
        bank_ids=np.asarray([source_map[tuple(map(int,k))] for k in train['key'][:nsmoke]],np.int64)
        bank = {k: v[bank_ids] if isinstance(v, np.ndarray) and len(v) == 4516 else v for k, v in bank.items()}
        bank_input = {k: v[bank_ids] if isinstance(v, np.ndarray) and len(v) == 4516 else v for k, v in bank_input.items()}
        train = {k: v[:nsmoke] if isinstance(v, np.ndarray) and len(v) == len(train['key']) else v for k, v in train.items()}
        learned = {k: v[:12] if isinstance(v, np.ndarray) and len(v) == 621 else v for k, v in learned.items()}
        unseen = {k: v[:12] if isinstance(v, np.ndarray) and len(v) == 619 else v for k, v in unseen.items()}
        dev = {k: v[:12] if isinstance(v, np.ndarray) and len(v) == len(dev['key']) else v for k, v in dev.items()}
    # Target and candidate values: fixed 10-step stored DCT3 playback, no fit/rotation/scale.
    cand_actions = key2._candidate_window_actions(np.arange(len(bank['length'])), bank)
    if cand_actions.shape[1:] != (10, 7): raise AssertionError('candidate action convention changed')
    # key2's train dictionary is ordered exactly like the beginning of bank_input only when no heldout
    # learned rows are omitted. Map train keys back to global candidate ids exactly.
    bank_map = {tuple(map(int,k)):i for i,k in enumerate(bank_input['key'])}
    train_ids = np.asarray([bank_map[tuple(map(int,k))] for k in train['key']], np.int64)
    train_actions = np.asarray(train['actions'], np.float32)
    q_train_vlm = make_observation_matrix(train)
    q_dev_vlm = make_observation_matrix(dev)
    q_learn_vlm = make_observation_matrix(learned)
    q_unseen_vlm = make_observation_matrix(unseen)
    # Bank query keys can use exact start/end image information; action is excluded for IDM.
    # Unit end observations are gathered only from exact feature keys (no temporal interpolation).
    end_keys = np.asarray(bank_input['key']).copy()
    end_keys[:, 2] = np.asarray(bank['source'])[:, 3] - 1
    feat_store = key2.ExactFeatureStore(run_root, missing_root, seen_stride5, unseen_stride5)
    try:
        end_tok, end_grp = feat_store.get(end_keys, domain='libero40')
        end_pool = pooled_tokens(end_tok, end_grp)
        endpoint_cache = {'exact': True, 'source': 'existing exact feature stores', 'rows': len(end_keys)}
    except KeyError:
        if unit_end_features is None:
            raise RuntimeError('IDM needs exact unit-end observations; run --extract-unit-ends and retry')
        end_pool = exact_pool_with_extra(end_keys, feat_store, unit_end_features)
        endpoint_cache = {'exact': True, 'source': str(unit_end_features), 'rows': len(end_keys)}
    unit_cont = make_unit_input(bank, bank_input, use_action=True, end_observation=end_pool)
    unit_idm = make_unit_input(bank, bank_input, use_action=False, end_observation=end_pool)
    # Use the bank subset which excludes held-out learned evaluation demos for every fit.
    learned_pairs = set(zip(learned['key'][:,0].tolist(), learned['key'][:,1].tolist()))
    fit_candidate_ids = np.asarray([i for i,k in enumerate(bank_input['key']) if (int(k[0]),int(k[1])) not in learned_pairs], np.int64)
    if not np.array_equal(train_ids, fit_candidate_ids):
        raise AssertionError('fit query order/candidate order differ; training targets cannot be aligned safely')
    if set(zip(bank['task'][fit_candidate_ids].tolist(), bank['episode'][fit_candidate_ids].tolist())) & set(zip(learned['task'].tolist(), learned['episode'].tolist())):
        raise AssertionError('learned eval demos leaked into fitting candidate pool')
    fit_actions = cand_actions[fit_candidate_ids]
    fit_unit_cont = unit_cont[fit_candidate_ids]; fit_unit_idm = unit_idm[fit_candidate_ids]
    fit_task = bank['task'][fit_candidate_ids]; fit_ep = bank['episode'][fit_candidate_ids]
    fit_error_matrix_np=key2._arm6_candidate_mse(train_actions,fit_actions)
    split_error_matrices={name:key2._arm6_candidate_mse(split['actions'],cand_actions)
        for name,split in (('dev',dev),('learned',learned),('unseen',unseen))}
    fit_error_matrix=torch.as_tensor(fit_error_matrix_np,device=device)
    qfit = q_train_vlm
    # Build VQ codebooks using only memory units permitted in fitting.
    models: dict[str, Any] = {}; selections=[]
    vq_info={}
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Full requested Dev grid: two learning rates × {4,8,16} epochs.
    # Smoke intentionally uses one setting to keep the integration check short.
    cfgs = ([{'id': f'lr{lr:g}-ep{ep}', 'lr':lr, 'epochs':ep}
             for lr in LEARNING_RATES for ep in EPOCHS] if not smoke else
            [{'id': f'lr{learning_rate:g}-ep{epochs}', 'lr':learning_rate, 'epochs':epochs}])
    for cfg in cfgs:
        for n_codes in codes:
            vq, info = train_vqvae(fit_actions, n_codes, epochs=cfg['epochs'],
                                    learning_rate=cfg['lr'], device=device, seed=SEED)
            vq_info[f'{n_codes}-{cfg["id"]}'] = info
            with torch.no_grad():
                code_ids, _, _, _ = vq(torch.as_tensor(fit_actions[..., :6],device=device,dtype=torch.float32))
            code_labels = code_ids.cpu().numpy()
            with torch.no_grad():
                all_code_labels=vq(torch.as_tensor(cand_actions[..., :6],device=device,dtype=torch.float32))[0].cpu().numpy()
            labels = code_labels
            # VQ shared query network + unit-key ranking; classifier is fit on train units only.
            qnet=QueryEncoder(qfit.shape[-1]).to(device)
            code_head=_fit_code_head(qnet, qfit, labels, epochs=cfg['epochs'], learning_rate=cfg['lr'],device=device,seed=SEED)
            knet=UnitKeyEncoder(fit_unit_cont.shape[-1]).to(device)
            # Frozen query for candidate retrieval, action keys are trained to align to its query latent.
            qnet.eval(); qfit_t=_as_t(qfit,device).detach()
            key_input=_as_t(fit_unit_cont,device); opt=torch.optim.AdamW(knet.parameters(),lr=cfg['lr'])
            for _ in range(cfg['epochs']):
                opt.zero_grad(set_to_none=True); krep=knet(key_input); qrep=qnet(qfit_t)
                # Code agreement remains the primary constraint; retrieval ordering learns a fine choice within code.
                scr=qrep@krep.T*qnet.logit_scale.exp().clamp(max=100)
                errs=fit_error_matrix
                al=torch.as_tensor(exclude_self_episode(train['task'],train['episode'],fit_task,fit_ep),device=device)
                loss=listwise_loss(scr,errs,al)
                loss.backward();opt.step()
            # Re-evaluate VQ class head and candidate-key embeddings on Dev.
            # Score each candidate by action-key dot product, restricted to predicted code.
            qnet_scores=[]
            krep_all=_encode_keys(knet,unit_cont,device)
            learned_scale=float(qnet.logit_scale.detach().exp().clamp(max=100).cpu())
            blend_temperatures=None;rank_temperature=None
            for nsplit, split, qx in [('dev',dev,q_dev_vlm),('learned',learned,q_learn_vlm),('unseen',unseen,q_unseen_vlm)]:
                qrep=_predict_latents(qnet,qx,device)
                # The code head is trained on the query latent, not raw observation features.
                with torch.no_grad(): logits=code_head(_as_t(qrep,device)).cpu().numpy()
                pred=logits.argmax(1); scr=qrep@krep_all.T*learned_scale
                allowed_eval=exclude_self_episode(split['task'],split['episode'],bank['task'],bank['episode'])
                code_allowed=(all_code_labels[None,:] == pred[:,None]) & allowed_eval
                no_code_candidate=~code_allowed.any(axis=1)
                scr[~code_allowed]=-np.inf
                if no_code_candidate.any():
                    full=qrep[no_code_candidate]@krep_all.T
                    full[~allowed_eval[no_code_candidate]]=-np.inf
                    scr[no_code_candidate]=full
                # The fit pool has fewer candidates; map to fitting action-error matrix.
                err=split_error_matrices[nsplit]
                top=scr.argmax(1)
                mse=action_error(cand_actions[top],split['actions'])
                if nsplit=='dev':
                    blend_temperatures=select_blend_temperatures(scr,cand_actions,split['actions'])
                    rank_temperature=select_rank_temperature(scr,err)['temperature']
                block={'scores':scr,'errors':err,'top1_indices':top,'retrieved_actions':cand_actions[top],
                       'top1_actions':_bootstrap_summary(mse,split),
                       'ranking':_rank_by_slice(scr,err,split,rank_temperature),'top1_indices':top,
                       'rank_score_temperature':rank_temperature}
                with torch.no_grad(): decoded=vq.decode_code(torch.as_tensor(pred,device=device)).cpu().numpy()
                block['decoded_actions']=decoded
                block['decoded_mse']=_bootstrap_summary(action_error(decoded,split['actions']),split)
                block['topk_blend_mse']={};block['topk_blend_temperatures']={}
                for kk in (1,5,20):
                    temp_cfg=blend_temperatures[str(kk)]
                    blended,_=topk_blend(scr,cand_actions,kk,temperature=temp_cfg['temperature'])
                    if kk==1:block['retrieved_actions']=blended
                    block['topk_blend_mse'][str(kk)]=_bootstrap_summary(action_error(blended,split['actions']),split)
                    block['topk_blend_temperatures'][str(kk)]=temp_cfg
                qnet_scores.append((nsplit,block))
                if nsplit=='dev':
                    selections.append({'config_id':f'vq{n_codes}-{cfg["id"]}','dev_top1_arm6_mse':float(mse.mean()),'vq_codes':n_codes})
            models[f'vq{n_codes}-{cfg["id"]}']={'vq':vq,'qnet':qnet,'code_head':code_head,'knet':knet,'metrics':qnet_scores}
    # Continuous retrieval and decoder share their trained query latent. The
    # decoder is attached only after retrieval representation fit.
    for cfg in cfgs:
      for latent_name, kinput, use_action in (('cont',fit_unit_cont,True),('idm',fit_unit_idm,False)):
        _seed_everything(SEED + cfg['epochs'] + int(cfg['lr'] * 1e6) + (17 if latent_name=='idm' else 0))
        qnet=QueryEncoder(qfit.shape[-1]).to(device);knet=UnitKeyEncoder(kinput.shape[-1]).to(device)
        qin=_as_t(qfit,device);kin=_as_t(kinput,device)
        errs=fit_error_matrix
        allowed=torch.as_tensor(exclude_self_episode(train['task'],train['episode'],fit_task,fit_ep),device=device)
        opt=torch.optim.AdamW(list(qnet.parameters())+list(knet.parameters()),lr=cfg['lr'])
        logs=[]
        for ep in range(cfg['epochs']):
            q=qnet(qin);k=knet(kin)
            scores=q@k.T*qnet.logit_scale.exp().clamp(max=100)
            loss=listwise_loss(scores,errs,allowed)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(list(qnet.parameters())+list(knet.parameters()),1.);opt.step()
            logs.append({'epoch':ep+1,'loss':float(loss.detach())})
        decoder=ActionDecoder().to(device) if latent_name=='cont' else None
        if decoder is not None:
            # Fit a decoder over the same frozen q representation, leaving retrieval unchanged.
            qnet.eval(); qvals=_as_t(qfit,device).detach(); optd=torch.optim.AdamW(decoder.parameters(),lr=learning_rate)
            rng=np.random.default_rng(SEED+1);order=np.arange(len(qfit))
            for _ in range(cfg['epochs']):
                rng.shuffle(order)
                for ix0 in np.array_split(order,max(1,int(np.ceil(len(order)/64)))):
                    ix=torch.as_tensor(ix0,device=device)
                    # The decoder predicts the arm6 representation; gripper is
                    # copied from truth only in the scoring metric.
                    loss=F.mse_loss(decoder(qnet(qvals[ix])),_as_t(train_actions[..., :6],device)[ix])
                    optd.zero_grad(set_to_none=True);loss.backward();optd.step()
        metrics=[]
        krep_all=_encode_keys(knet,unit_cont if latent_name=='cont' else unit_idm,device)
        learned_scale=float(qnet.logit_scale.detach().exp().clamp(max=100).cpu())
        blend_temperatures=None;rank_temperature=None
        for split_name,split,qx in [('dev',dev,q_dev_vlm),('learned',learned,q_learn_vlm),('unseen',unseen,q_unseen_vlm)]:
            qrep=_predict_latents(qnet,qx,device);scores=qrep@krep_all.T*learned_scale
            allowed_eval=exclude_self_episode(split['task'],split['episode'],bank['task'],bank['episode'])
            scores[~allowed_eval]=-np.inf
            err=split_error_matrices[split_name];idx=scores.argmax(1)
            retrieval_mse=action_error(cand_actions[idx],split['actions'])
            if split_name=='dev':
                blend_temperatures=select_blend_temperatures(scores,cand_actions,split['actions'])
                rank_temperature=select_rank_temperature(scores,err)['temperature']
            block={'scores':scores,'errors':err,'top1_indices':idx,'retrieved_actions':cand_actions[idx],
                'top1_actions':_bootstrap_summary(retrieval_mse,split),
                'ranking':_rank_by_slice(scores,err,split,rank_temperature),
                'rank_score_temperature':rank_temperature}
            if decoder is not None:
                with torch.no_grad(): pred=decoder(_as_t(qrep,device)).cpu().numpy()
                block['decoded_actions']=pred
                block['decoded_mse']=_bootstrap_summary(action_error(pred,split['actions']),split)
            block['topk_blend_mse']={};block['topk_blend_temperatures']={}
            for kk in (1,5,20):
                temp_cfg=blend_temperatures[str(kk)]
                blended,_=topk_blend(scores,cand_actions,kk,temperature=temp_cfg['temperature'])
                if kk==1:block['retrieved_actions']=blended
                block['topk_blend_mse'][str(kk)]=_bootstrap_summary(action_error(blended,split['actions']),split)
                block['topk_blend_temperatures'][str(kk)]=temp_cfg
            metrics.append((split_name,block))
            if split_name=='dev': selections.append({'config_id':f'la-{latent_name}-{cfg["id"]}',
                'dev_top1_arm6_mse':float(retrieval_mse.mean()),'latent':latent_name})
        models[f'la-{latent_name}-{cfg["id"]}']={'qnet':qnet,'knet':knet,'decoder':decoder,'metrics':metrics,'training':logs}
    # Required input comparison: V-JEPA + VLM + history + state versus VLM-only.
    qfit_vj=make_observation_matrix(train,vjepa=True)
    q_dev_vj=make_observation_matrix(dev,vjepa=True)
    q_learn_vj=make_observation_matrix(learned,vjepa=True)
    q_unseen_vj=make_observation_matrix(unseen,vjepa=True)
    unit_cont_vj=make_unit_input(bank,bank_input,use_action=True,use_vjepa=True,end_observation=end_pool)
    fit_unit_cont_vj=unit_cont_vj[fit_candidate_ids]
    for cfg in cfgs:
        _seed_everything(SEED + cfg['epochs'] + int(cfg['lr'] * 1e6) + 101)
        qnet=QueryEncoder(qfit_vj.shape[-1]).to(device);knet=UnitKeyEncoder(fit_unit_cont_vj.shape[-1]).to(device)
        qin=_as_t(qfit_vj,device);kin=_as_t(fit_unit_cont_vj,device)
        errs=fit_error_matrix
        allowed=torch.as_tensor(exclude_self_episode(train['task'],train['episode'],fit_task,fit_ep),device=device)
        opt=torch.optim.AdamW(list(qnet.parameters())+list(knet.parameters()),lr=cfg['lr'])
        logs=[]
        for ep in range(cfg['epochs']):
            q=qnet(qin);k=knet(kin)
            scores=q@k.T*qnet.logit_scale.exp().clamp(max=100)
            loss=listwise_loss(scores,errs,allowed)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(list(qnet.parameters())+list(knet.parameters()),1.);opt.step()
            logs.append({'epoch':ep+1,'loss':float(loss.detach())})
        metrics=[]
        krep_all=_encode_keys(knet,unit_cont_vj,device)
        learned_scale=float(qnet.logit_scale.detach().exp().clamp(max=100).cpu())
        blend_temperatures=None;rank_temperature=None
        for split_name,split,qx in [('dev',dev,q_dev_vj),('learned',learned,q_learn_vj),('unseen',unseen,q_unseen_vj)]:
            qrep=_predict_latents(qnet,qx,device);scores=qrep@krep_all.T*learned_scale
            scores[~exclude_self_episode(split['task'],split['episode'],bank['task'],bank['episode'])]=-np.inf
            err=split_error_matrices[split_name];idx=scores.argmax(1)
            mse=action_error(cand_actions[idx],split['actions'])
            if split_name=='dev':
                blend_temperatures=select_blend_temperatures(scores,cand_actions,split['actions'])
                rank_temperature=select_rank_temperature(scores,err)['temperature']
            block={'scores':scores,'errors':err,'top1_indices':idx,'retrieved_actions':cand_actions[idx],
                'top1_actions':_bootstrap_summary(mse,split),
                'ranking':_rank_by_slice(scores,err,split,rank_temperature),
                'rank_score_temperature':rank_temperature,'topk_blend_mse':{},'topk_blend_temperatures':{}}
            for kk in (1,5,20):
                temp_cfg=blend_temperatures[str(kk)]
                blended,_=topk_blend(scores,cand_actions,kk,temperature=temp_cfg['temperature'])
                if kk==1:block['retrieved_actions']=blended
                block['topk_blend_mse'][str(kk)]=_bootstrap_summary(action_error(blended,split['actions']),split)
                block['topk_blend_temperatures'][str(kk)]=temp_cfg
            metrics.append((split_name,block))
            if split_name=='dev':selections.append({'config_id':f'la-cont-vjepa-{cfg["id"]}',
                'dev_top1_arm6_mse':float(mse.mean()),'latent':'continuous','input':'vjepa+vlm+history+state'})
        models[f'la-cont-vjepa-{cfg["id"]}']={'qnet':qnet,'knet':knet,'decoder':None,'metrics':metrics,'training':logs}
    def family_of(row):
        cid=str(row.get('config_id',''))
        if cid.startswith('vq64-'): return 'vq64'
        if cid.startswith('vq256-'): return 'vq256'
        if cid.startswith('la-cont-vjepa-'): return 'cont_vjepa'
        if cid.startswith('la-idm-'): return 'idm'
        if cid.startswith('la-cont-'): return 'cont'
        return cid.split('-')[0]
    selected_by_family={}
    for family in sorted({family_of(x) for x in selections}):
        group=[x for x in selections if family_of(x)==family]
        if group: selected_by_family[family]=choose_dev(group)
    selected=choose_dev(selections)
    # Save only model tensors/state and all query-level outputs; report scores as method × split.
    for name,obj in models.items():
        state={'qnet':{k:v.detach().cpu() for k,v in obj['qnet'].state_dict().items()},
               'knet':{k:v.detach().cpu() for k,v in obj['knet'].state_dict().items()},'config':name}
        if obj.get('decoder') is not None: state['decoder']={k:v.detach().cpu() for k,v in obj['decoder'].state_dict().items()}
        if obj.get('code_head') is not None: state['code_head']={k:v.detach().cpu() for k,v in obj['code_head'].state_dict().items()}
        if obj.get('vq') is not None: state['vq']={k:v.detach().cpu() for k,v in obj['vq'].state_dict().items()}
        torch.save(state,output/f'{name}.pt')
        for split_name,block in obj['metrics']:
            np.savez_compressed(output/f'{name}-{split_name}-predictions.npz',
                top1_indices=np.asarray(block['top1_indices']),
                candidate_ids=np.arange(len(cand_actions),dtype=np.int64),
                retrieved_actions=np.asarray(block.get('retrieved_actions',np.zeros((0,10,7))),np.float32),
                scores=np.asarray(block['scores'],np.float32),
                errors=np.asarray(block['errors'],np.float32),
                decoded_actions=np.asarray(block.get('decoded_actions',np.zeros((0,10,6))),np.float32))
    results={'status':'complete','seed':SEED,'device':device,'memory_fraction':memory_fraction,
        'fixed_counts':{'units':len(bank['length']),'learned':len(learned['key']),'unseen':len(unseen['key']),'dev':len(dev['key'])},
        'fit_candidate_count':len(fit_candidate_ids),'endpoint_observation_cache':endpoint_cache,
        'target_action_inputs_excluded':True,'hyperparameter_selection':'Dev only; rank by top-1 arm6 playback MSE',
        'selected_dev_config':selected,'selected_dev_by_family':selected_by_family,
        'vqvae':vq_info,'selection_candidates':selections,
        'methods':{name:{split:{'top1_arm6':block['top1_actions'],'ranking':block['ranking'],
            'topk_blend_arm6':block.get('topk_blend_mse'),
            'topk_blend_temperatures':block.get('topk_blend_temperatures'),
            'rank_score_temperature':block.get('rank_score_temperature'),
            **({'decoded_arm6':block['decoded_mse']} if 'decoded_mse' in block else {})}
            for split,block in obj['metrics']} for name,obj in models.items()},
        'ranking_metric_definition':'KEY2-compatible: teacher softmax(-arm6_MSE/.01), score softmax(score/Dev-selected temperature), graded Recall@{1,5,20}, gain exp(-MSE/.01) NDCG@20, MRR of exact best candidate; masks excluded candidates.',
        'key2_fixed_baselines': json.loads(Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-27-production/fixed-baselines/baselines.json').read_text())
            if Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-27-production/fixed-baselines/baselines.json').exists() else None,
        'key2_full_results': json.loads(Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-27-production/results.json').read_text())
            if Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-27-production/results.json').exists() else None,
        'selection_masks':'same-episode excluded; fixed KEY2 bank; action labels used only in teacher/dev scoring',
        'v_jepa_query_variant':'trained V-JEPA+VLM+history+state continuous retrieval variant; unit key includes exact V-JEPA start observation'}
    (output/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    lines=['# LA1: 潜在出口の比較','','|手法|潜在|出口|Learned top-1 all/hard|Learned top-5/20 blend|Unseen top-1 all/hard|Unseen top-5/20 blend|','|---|---|---|---:|---:|---:|---:|']
    for name,obj in models.items():
        latent='VQ' if name.startswith('vq') else ('IDM' if 'idm' in name else '連続')
        for exit_name in (('dec','ret') if name.startswith('vq') else (('ret','dec') if 'la-cont-' in name and 'vjepa' not in name else ('ret',))):
            def value(split_name):
                block=dict(obj['metrics'])[split_name]
                sm=block['decoded_mse'] if exit_name=='dec' and 'decoded_mse' in block else block['top1_actions']
                return f"{sm['mean']:.5f}/{sm['hard']['mean'] if sm['hard']['mean'] is not None else float('nan'):.5f}"
            def blend(split_name,kk):
                block=dict(obj['metrics'])[split_name].get('topk_blend_mse',{}).get(str(kk))
                return '—' if block is None else f"{block['mean']:.5f}/{block['hard']['mean'] if block['hard']['mean'] is not None else float('nan'):.5f}"
            bm=lambda split_name: ('— / —' if exit_name=='dec' else f"{blend(split_name,5)} / {blend(split_name,20)}")
            lines.append(f"|{name}|{latent}|{exit_name}|{value('learned')}|{bm('learned')}|{value('unseen')}|{bm('unseen')}|")
    baseline_doc=results.get('key2_fixed_baselines')
    if baseline_doc:
        lines+=['','## KEY2-F 固定比較基準','',
            '|基準|Learned all/hard|Unseen all/hard|','|---|---:|---:|']
        for label,field in [('π0.5','pi05'),('直前保持','hold_last'),('ゼロ','zero'),
                            ('random-5','random5'),('oracle playback','oracle')]:
            vals=[]
            for split_name in ('learned','unseen'):
                block=baseline_doc['metrics'][split_name]
                if field=='random5': allv=block['random5_actions_all']['mse_gripper_truth'];hardv=block['random5_actions_hard']['mse_gripper_truth']
                elif field=='oracle': allv=block['oracle_playback_all']['mse_gripper_truth'];hardv=block['oracle_playback_hard']['mse_gripper_truth']
                else:
                    allv=block['baselines'][field]['mse_gripper_truth'];hardv=block['baselines_hard'][field]['mse_gripper_truth']
                vals.append(f'{allv:.5f}/{hardv:.5f}')
            lines.append(f'|{label}|{vals[0]}|{vals[1]}|')
        lines+=['','KEY2-F の共有 E / P5 完了結果は `key2_full_results` を参照。取得時点で未完了なら統合表への追加を保留し、推測値は記載しない。']
    lines+=['','設定選択は Dev（LIBERO-90 task 6/7）だけで行った。Learned/Eval の正解行動は学習・選択に使わず、採点にだけ使用。',
        'VQ のコードは固定長10ステップの腕6軸行動から学習した。グリッパーは採点時に正解を使用。',
        '注：この実行は保存単位・連続潜在の基本経路を記録する。KEY2-F の最終ベースライン表が完成後、同じ行定義の値を統合する。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return results


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-root',type=Path,default=Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01'))
    p.add_argument('--missing-root',type=Path,default=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-02/prefix-missing'))
    p.add_argument('--seen-stride5',type=Path,default=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/seg5-cache'))
    p.add_argument('--unseen-stride5',type=Path,default=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/unseen-stride5'))
    p.add_argument('--vjepa-eval-cache',type=Path,default=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-27-production/vjepa/libero90_eval'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--memory-fraction',type=float,default=.18)
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--codes',type=int,nargs='+',default=list(VQ_CODES));p.add_argument('--smoke',action='store_true')
    p.add_argument('--extract-unit-ends',action='store_true',help='extract exact missing unit-end π0.5 prefix tokens')
    p.add_argument('--unit-end-features',type=Path,default=Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/latent-action-01/unit-end-features'))
    a=p.parse_args()
    if a.extract_unit_ends:
        bank,bank_input,*_=key2._build_key2_queries(a.run_root,a.missing_root,a.seen_stride5,a.unseen_stride5,a.vjepa_eval_cache)
        keys=np.asarray(bank_input['key']).copy();keys[:,2]=np.asarray(bank['source'])[:,3]-1
        rep=extract_unit_end_prefixes(keys,a.unit_end_features,run_root=a.run_root,
            primary_cache=a.seen_stride5,secondary_cache=a.run_root,extra_caches=(a.missing_root,))
        print(json.dumps(rep,indent=2));return
    result=run_experiment(a.output,run_root=a.run_root,missing_root=a.missing_root,
        seen_stride5=a.seen_stride5,unseen_stride5=a.unseen_stride5,vjepa_eval_cache=a.vjepa_eval_cache,
        device=a.device,memory_fraction=a.memory_fraction,epochs=a.epochs,
        learning_rate=a.learning_rate,codes=tuple(a.codes),smoke=a.smoke,unit_end_features=a.unit_end_features)
    print(json.dumps({'status':result['status'],'output':str(a.output),'selected':result['selected_dev_config']}))


if __name__=='__main__': main()
