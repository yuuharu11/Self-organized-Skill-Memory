"""CLIP-style query/skill embedding retrieval study for intermediate actions."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .abstracted_m1 import (
    _decode_residual_draws, _fit_abstracted_regressor, _mse,
    _predict_abstracted, _prepare_m1_train,
)
from .splits import SPLITS, subset_rows as _subset
from .compare_splits import (
    QUERY_MODES, RETRIEVAL_T0S, _same_class_alternatives,
    _predict_xhat0,
)
from .experiment import (
    DEFAULT_CACHE, DEFAULT_DATASET, DEFAULT_OUTPUT_ROOT, HORIZONS,
    _episode_train_validation_indices, _onehot, _query_features, _query_key,
    _standardize, _standardize_fit, integrate_flow_sources, load_rows,
    make_units, nearest_units, retrieve_top1, retrieve_top1_same_class,
    train_flow,
)
from ..methods.m6 import M6
from ....research_paths import REPO_ROOT, STATE_DIR

COMPONENT_DIM = 32
EMBED_DIM = 3 * COMPONENT_DIM
T0_ACTION = 0.2
EVAL_T0S = (0.2, 0.4, 0.6, 0.8)
CLIP_EPOCHS = 35
CLIP_PATIENCE = 6
CLIP_BATCH = 256
DEFAULT_PI05_BASE_PARAMS = STATE_DIR / "checkpoints/openpi-assets/checkpoints/pi05_base/params"
DEFAULT_PALIGEMMA_TOKENIZER = STATE_DIR / "cache/skillnet/big_vision/paligemma_tokenizer.model"
DEFAULT_ANNOTATIONS = REPO_ROOT / "skill_moe/skillnet/examples/libero/annotations/instruct2plan_40.json"


def plan_text_for_segment(task_name, segment_id, annotations):
    """Map a cached segment id to the short π0.5-style plan phrase."""
    segment_id = int(segment_id)
    if segment_id == -1:
        return ""
    try:
        plan = annotations[task_name]["plan"]
    except KeyError as exc:
        raise KeyError(f"no plan annotation for task {task_name!r}") from exc
    if segment_id < 0 or segment_id >= len(plan):
        raise ValueError(f"segment {segment_id} is outside task plan with {len(plan)} steps: {task_name}")
    return str(plan[segment_id])


def attach_subtask_texts(rows, *, cache_dir, annotations_path):
    """Join per-row segment ids with task plans, preserving every cache row."""
    manifest = json.loads((Path(cache_dir) / "manifest.json").read_text())
    annotations = json.loads(Path(annotations_path).read_text())
    task_by_episode = {str(item["episode_id"]): str(item["task"])
                       for item in manifest["episodes"]}
    texts = []
    for episode_id, segment_id in zip(rows["episode_id"], rows["skill_segment"]):
        episode_id = str(episode_id)
        if episode_id not in task_by_episode:
            raise KeyError(f"episode {episode_id} is absent from feature-cache manifest")
        texts.append(plan_text_for_segment(task_by_episode[episode_id], segment_id, annotations))
    if len(texts) != len(rows["episode_id"]):
        raise AssertionError("subtask text join changed the cache row count")
    result = dict(rows)
    result["subtask_text"] = np.asarray(texts, dtype=str)
    return result


def mean_token_embedding(text, tokenizer, token_embedding):
    """Mean frozen PaliGemma token embedding; empty/-1 subtask maps to zero."""
    text = str(text)
    if not text:
        return np.zeros((token_embedding.shape[-1],), dtype=np.float32)
    token_ids = tokenizer.encode(text, add_bos=False)
    if not token_ids:
        return np.zeros((token_embedding.shape[-1],), dtype=np.float32)
    return np.asarray(token_embedding[np.asarray(token_ids, dtype=np.int64)],
                      dtype=np.float32).mean(axis=0)


def frozen_pi05_plan_embeddings(texts, *, params_path=DEFAULT_PI05_BASE_PARAMS,
                                tokenizer_path=DEFAULT_PALIGEMMA_TOKENIZER):
    """Average local π0.5 base token embeddings without downloading weights."""
    params_path, tokenizer_path = Path(params_path), Path(tokenizer_path)
    if not params_path.exists():
        raise FileNotFoundError(f"local π0.5 base checkpoint not found: {params_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"local PaliGemma tokenizer not found: {tokenizer_path}")
    import sentencepiece
    from openpi.models import model as openpi_model

    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    params = openpi_model.restore_params(params_path, restore_type=np.ndarray)
    token_embedding = params["PaliGemma"]["llm"]["embedder"]["input_embedding"]
    if token_embedding.shape != (257152, 2048):
        raise ValueError(f"unexpected π0.5 PaliGemma token embedding shape: {token_embedding.shape}")
    result = np.stack([mean_token_embedding(text, tokenizer, token_embedding) for text in texts])
    del params, token_embedding
    return result.astype(np.float32)


def split_suite_mask(task_indices, train_task_indices):
    """Return only training-suite rows; held-out query suites cannot enter fit."""
    return np.isin(np.asarray(task_indices), np.asarray(train_task_indices))


class DualEncoder(nn.Module):
    def __init__(self, feature_dim, key_dim, state_dim, base_dim, num_classes,
                 *, use_subtask: bool | None = None, subtask_mode: str | None = None,
                 subtask_input_dim: int = COMPONENT_DIM,
                 embed_dim: int = EMBED_DIM):
        super().__init__()
        if subtask_mode is None:
            subtask_mode = "verb_class" if use_subtask else "none"
        if subtask_mode not in ("none", "verb_class", "sentence"):
            raise ValueError(f"unknown subtask mode {subtask_mode!r}")
        self.subtask_mode = subtask_mode
        self.use_subtask = subtask_mode != "none"
        self.class_embedding = nn.Embedding(num_classes, 32) if subtask_mode == "verb_class" else None
        self.image_tower = nn.Sequential(
            nn.Linear(feature_dim + key_dim, 128), nn.GELU(),
            nn.Linear(128, COMPONENT_DIM))
        self.state_tower = nn.Sequential(
            nn.Linear(state_dim, 64), nn.GELU(), nn.Linear(64, COMPONENT_DIM))
        subtask_input_dim = 32 if subtask_mode == "verb_class" else subtask_input_dim
        self.subtask_tower = (nn.Sequential(nn.Linear(subtask_input_dim, 64), nn.GELU(),
                                            nn.Linear(64, COMPONENT_DIM))
                              if self.use_subtask else None)
        self.w_img = nn.Parameter(torch.tensor(1.0))
        self.w_state = nn.Parameter(torch.tensor(1.0))
        self.w_sub = nn.Parameter(torch.tensor(1.0)) if self.use_subtask else None
        self.skill = nn.Sequential(nn.Linear(base_dim, 128), nn.GELU(),
                                   nn.Linear(128, embed_dim))
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / 0.07), dtype=torch.float32))

    def encode_query(self, subtask_input, feature, key, state, *,
                     include_img=True, include_state=True, include_subtask=True):
        e_img = F.normalize(self.image_tower(torch.cat([feature, key], dim=-1)), dim=-1)
        e_state = F.normalize(self.state_tower(state), dim=-1)
        if self.subtask_mode == "verb_class":
            subtask_features = self.class_embedding(subtask_input.long())
            e_sub = F.normalize(self.subtask_tower(subtask_features), dim=-1)
        elif self.subtask_mode == "sentence":
            e_sub = F.normalize(self.subtask_tower(subtask_input.float()), dim=-1)
            has_sentence = (subtask_input.abs().sum(dim=-1, keepdim=True) > 0)
            e_sub = torch.where(has_sentence, e_sub, torch.zeros_like(e_sub))
        else:
            e_sub = torch.zeros_like(e_img)
        if not include_img:
            e_img = torch.zeros_like(e_img)
        if not include_state:
            e_state = torch.zeros_like(e_state)
        if not include_subtask or not self.use_subtask:
            e_sub = torch.zeros_like(e_sub)
        joined = torch.cat([self.w_img * e_img, self.w_state * e_state,
                            self.w_sub * e_sub if self.use_subtask else e_sub], dim=-1)
        return F.normalize(joined, dim=-1)

    def modality_weights(self):
        return {"image": float(self.w_img.detach().cpu()),
                "state": float(self.w_state.detach().cpu()),
                "subtask": float(self.w_sub.detach().cpu()) if self.w_sub is not None else 0.0}

    def encode_skill(self, base):
        return F.normalize(self.skill(base), dim=-1)

    def temperature(self):
        return self.logit_scale.exp().reciprocal().clamp(0.01, 1.0)


class FrozenEmbeddingMemory:
    """Append-only values; its API has no references to trainable tower params."""
    def __init__(self, embed_dim=EMBED_DIM):
        self.query_embeddings = np.empty((0, embed_dim), dtype=np.float32)
        self.skill_embeddings = np.empty((0, embed_dim), dtype=np.float32)
        self.bases = np.empty((0, 0), dtype=np.float32)

    def add(self, query_embeddings, skill_embeddings, bases):
        query_embeddings = np.asarray(query_embeddings, np.float32)
        skill_embeddings = np.asarray(skill_embeddings, np.float32)
        bases = np.asarray(bases, np.float32)
        if query_embeddings.shape != skill_embeddings.shape:
            raise ValueError("query and skill embedding shapes differ")
        self.query_embeddings = np.concatenate([self.query_embeddings, query_embeddings], axis=0)
        self.skill_embeddings = np.concatenate([self.skill_embeddings, skill_embeddings], axis=0)
        if self.bases.size == 0:
            self.bases = bases.copy()
        else:
            self.bases = np.concatenate([self.bases, bases], axis=0)


def _nearest_cross_episode_q05(bases, episode_ids, *, sample_count=2048, seed=0,
                               query_chunk=128, reference_chunk=2048, device):
    """q05 of sampled rows' nearest base distance to a different episode."""
    rng = np.random.default_rng(seed)
    n = len(bases)
    sampled = rng.choice(n, size=min(sample_count, n), replace=False)
    values = torch.as_tensor(bases, dtype=torch.float32, device=device)
    sample_values = values[sampled]
    episodes = np.asarray(episode_ids).astype(str)
    nearest = np.full(len(sampled), np.inf, np.float32)
    for q0 in range(0, len(sampled), query_chunk):
        q1 = min(len(sampled), q0 + query_chunk)
        query = sample_values[q0:q1]
        best = torch.full((q1 - q0,), float("inf"), device=device)
        for r0 in range(0, n, reference_chunk):
            r1 = min(n, r0 + reference_chunk)
            dist = torch.cdist(query, values[r0:r1])
            same_episode = episodes[sampled[q0:q1], None] == episodes[None, r0:r1]
            dist.masked_fill_(torch.as_tensor(same_episode, device=device), float("inf"))
            best = torch.minimum(best, dist.min(dim=1).values)
        nearest[q0:q1] = best.cpu().numpy()
    finite = nearest[np.isfinite(nearest)]
    if not len(finite):
        raise ValueError("no cross-episode base neighbors available")
    return float(np.quantile(finite, 0.05)), nearest


def _positive_mask(batch_b, batch_episode, distance_threshold, *, enabled):
    n = len(batch_b)
    positives = torch.eye(n, dtype=torch.bool, device=batch_b.device)
    if enabled:
        distance = torch.cdist(batch_b, batch_b)
        cross_episode = batch_episode[:, None] != batch_episode[None, :]
        positives |= (distance <= distance_threshold) & cross_episode
    return positives


def _multi_positive_loss(logits, positives):
    all_lse = torch.logsumexp(logits, dim=1)
    pos_lse = torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)
    forward = (all_lse - pos_lse).mean()
    reverse_logits = logits.T
    reverse_pos = positives.T
    reverse = (torch.logsumexp(reverse_logits, dim=1)
               - torch.logsumexp(reverse_logits.masked_fill(~reverse_pos, -torch.inf), dim=1)).mean()
    return 0.5 * (forward + reverse)


def fit_dual_encoder(train, bases, *, use_subtask: bool, nearby_positive: bool,
                     seed: int, epochs: int, patience: int, device,
                     allowed_task_indices=None, subtask_mode: str | None = None,
                     include_img: bool = True, include_state: bool = True):
    """Fit CLIP-style towers using only the supplied train-suite rows."""
    if allowed_task_indices is not None:
        unexpected = set(np.unique(train["task_index"]).tolist()) - set(allowed_task_indices)
        if unexpected:
            raise ValueError(f"query-tower training received non-training suites: {sorted(unexpected)}")
    if subtask_mode is None:
        subtask_mode = "verb_class" if use_subtask else "none"
    task_ids = np.unique(train["skill_class"])
    task_lookup = {int(label): i for i, label in enumerate(task_ids)} if subtask_mode == "verb_class" else {}
    if subtask_mode == "verb_class":
        subtask_values = np.asarray([task_lookup[int(label)] for label in train["skill_class"]], np.int64)
        subtask_input_dim = COMPONENT_DIM
    elif subtask_mode == "sentence":
        if "subtask_embedding" not in train:
            raise ValueError("sentence subtask mode requires frozen subtask_embedding rows")
        subtask_values = np.asarray(train["subtask_embedding"], dtype=np.float32)
        subtask_input_dim = int(subtask_values.shape[-1])
    elif subtask_mode == "none":
        subtask_values = np.zeros((len(train["a"]),), np.int64)
        subtask_input_dim = COMPONENT_DIM
    else:
        raise ValueError(f"unknown subtask mode {subtask_mode!r}")
    fit_ix, val_ix = _episode_train_validation_indices(train, seed + 19)
    f_mean, f_scale = _standardize_fit(train["f"][fit_ix])
    k_mean, k_scale = _standardize_fit(train["k"][fit_ix])
    s_mean, s_scale = _standardize_fit(train["s"][fit_ix])
    f_data = _standardize(train["f"], f_mean, f_scale)
    k_data = _standardize(train["k"], k_mean, k_scale)
    s_data = _standardize(train["s"], s_mean, s_scale)
    _, episode_codes = np.unique(train["episode_id"], return_inverse=True)
    threshold, sampled_nearest = _nearest_cross_episode_q05(
        bases[fit_ix], train["episode_id"][fit_ix], seed=seed + 31, device=device)

    ft = torch.as_tensor(f_data, dtype=torch.float32, device=device)
    kt = torch.as_tensor(k_data, dtype=torch.float32, device=device)
    st = torch.as_tensor(s_data, dtype=torch.float32, device=device)
    bt = torch.as_tensor(bases, dtype=torch.float32, device=device)
    ct = torch.as_tensor(subtask_values,
                         dtype=torch.long if subtask_mode in ("verb_class", "none") else torch.float32,
                         device=device)
    et = torch.as_tensor(episode_codes, dtype=torch.long, device=device)
    model = DualEncoder(ft.shape[-1], kt.shape[-1], st.shape[-1], bt.shape[-1],
                        len(task_ids), subtask_mode=subtask_mode,
                        subtask_input_dim=subtask_input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)
    generator = torch.Generator(device=device).manual_seed(seed + 401)
    best_state, best_loss, best_epoch, stale = None, float("inf"), 0, 0
    history = []
    for epoch in range(epochs):
        order = torch.as_tensor(fit_ix, device=device)
        order = order[torch.randperm(len(order), generator=generator, device=device)]
        model.train()
        train_losses = []
        for start in range(0, len(order), CLIP_BATCH):
            ix = order[start:start + CLIP_BATCH]
            q = model.encode_query(ct[ix], ft[ix], kt[ix], st[ix],
                                   include_img=include_img, include_state=include_state)
            h = model.encode_skill(bt[ix])
            scale = model.logit_scale.exp().clamp(1.0, 100.0)
            logits = scale * q @ h.T
            positive = _positive_mask(bt[ix], et[ix], threshold, enabled=nearby_positive)
            loss = _multi_positive_loss(logits, positive)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                model.logit_scale.clamp_(np.log(1.0), np.log(100.0))
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for start in range(0, len(val_ix), CLIP_BATCH):
                ix = torch.as_tensor(val_ix[start:start + CLIP_BATCH], device=device)
                q = model.encode_query(ct[ix], ft[ix], kt[ix], st[ix],
                                       include_img=include_img, include_state=include_state)
                h = model.encode_skill(bt[ix])
                logits = model.logit_scale.exp().clamp(1.0, 100.0) * q @ h.T
                positive = _positive_mask(bt[ix], et[ix], threshold, enabled=nearby_positive)
                valid_losses.append(float(_multi_positive_loss(logits, positive).cpu()))
        validation_loss = float(np.mean(valid_losses))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)),
                        "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_loss, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "use_subtask": subtask_mode != "none", "subtask_mode": subtask_mode,
        "include_img": bool(include_img), "include_state": bool(include_state),
        "positive_rule": "nearby_cross_episode_q05" if nearby_positive else "instance_only",
        "train_rows": int(len(fit_ix)), "validation_rows": int(len(val_ix)),
        "train_task_classes": task_ids.tolist(), "nearest_cross_episode_b_q05": threshold,
        "sampled_nearest_distance_q05": float(np.quantile(sampled_nearest, 0.05)),
        "epochs": len(history), "best_epoch": best_epoch,
        "best_validation_loss": best_loss, "final_validation_loss": history[-1]["validation_loss"],
        "temperature": float(model.temperature().cpu()), "history": history,
        "class_lookup": task_lookup, "normalizers": {"f_mean": f_mean, "f_scale": f_scale,
                         "k_mean": k_mean, "k_scale": k_scale, "s_mean": s_mean, "s_scale": s_scale},
    }


def _embed_rows(model, fit, rows, bases, *, device, batch_size=1024,
                include_img=True, include_state=True, include_subtask=True):
    norm = fit["normalizers"]
    q_out, h_out = [], []
    subtask_inputs = _subtask_inputs(rows, fit)
    with torch.no_grad():
        for start in range(0, len(bases), batch_size):
            stop = min(start + batch_size, len(bases))
            f = torch.as_tensor(_standardize(rows["f"][start:stop], norm["f_mean"], norm["f_scale"]), device=device)
            k = torch.as_tensor(_standardize(rows["k"][start:stop], norm["k_mean"], norm["k_scale"]), device=device)
            s = torch.as_tensor(_standardize(rows["s"][start:stop], norm["s_mean"], norm["s_scale"]), device=device)
            c = torch.as_tensor(subtask_inputs[start:stop],
                                dtype=torch.long if fit["subtask_mode"] in ("verb_class", "none") else torch.float32,
                                device=device)
            b = torch.as_tensor(bases[start:stop], device=device)
            q_out.append(model.encode_query(
                c, f, k, s, include_img=include_img, include_state=include_state,
                include_subtask=include_subtask).cpu().numpy())
            h_out.append(model.encode_skill(b).cpu().numpy())
    return np.concatenate(q_out), np.concatenate(h_out)


def _freeze_memory(model, fit, train, bases, *, device):
    """Snapshot both frozen towers for each stored training interval.

    ``query_embeddings`` are g(subtask, f, k, s) from the same query tower
    used at retrieval time; ``skill_embeddings`` are h(b). The memory stores
    both so symmetric g·g retrieval and the original g·h retrieval use the
    exact same fitted, frozen model.
    """
    q, h = _embed_rows(model, fit, train, bases, device=device)
    memory = FrozenEmbeddingMemory()
    memory.add(q, h, bases)
    return memory, q, h


def retrieve_embedding_top1(query_embeddings, memory_embeddings):
    """Return the item with maximum cosine similarity for normalized vectors."""
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    memory_embeddings = np.asarray(memory_embeddings, dtype=np.float32)
    if query_embeddings.ndim != 2 or memory_embeddings.ndim != 2:
        raise ValueError("query and memory embeddings must be rank-2")
    if query_embeddings.shape[1] != memory_embeddings.shape[1]:
        raise ValueError("query and memory embedding dimensions differ")
    if not len(memory_embeddings):
        raise ValueError("cannot retrieve from an empty memory")
    return (query_embeddings @ memory_embeddings.T).argmax(axis=1).astype(np.int64)


def _distribution_unit_sample(query_embedding, model, fit, centers, *, seed, device):
    with torch.no_grad():
        hc = model.encode_skill(torch.as_tensor(centers, dtype=torch.float32, device=device)).cpu().numpy()
    temperature = fit["temperature"]
    logits = query_embedding @ hc.T / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    sampled = np.asarray([rng.choice(len(centers), p=p) for p in probs], np.int64)
    return centers[sampled], sampled, probs


def _eval_actions(flow, flow_norm, m1, fiii, test, retrieved_indices, shuffled_indices,
                  retrieved_b, shuffled_b, random_b, oracle_b, retrieved_a, shuffled_a,
                  random_a, test_actions, *,
                  seed, t0, device):
    n = len(test_actions)
    test_h = dict(test)
    test_h["a"] = test_actions
    rng = np.random.default_rng(seed)
    noises = [rng.standard_normal((n, int(np.prod(test_actions.shape[1:])))).astype(np.float32)
              for _ in range(4)]
    raw_metrics, _ = integrate_flow_sources(
        flow, flow_norm, test_h, noises,
        {"retrieved": retrieved_a.reshape(n, -1),
         "shuffled": shuffled_a.reshape(n, -1),
         "random": random_a.reshape(n, -1),
         "oracle": test_actions.reshape(n, -1)},
        t0, device=device, return_per_row=True)
    iii = {}
    for name, b_values in {"retrieved": retrieved_b, "shuffled": shuffled_b,
                           "random": random_b, "oracle_b": oracle_b}.items():
        prediction = _predict_abstracted(m1, fiii, b_values, test["f"], test["s"],
                                         t0=t0, seed=seed + 100)
        iii[name] = _mse(prediction, test_actions)
    for key, value in raw_metrics.items():
        iii[f"ragdp_{key}"] = float(value)
    return iii


def _query_embedding(model, fit, rows, *, device, include_img=True,
                     include_state=True, include_subtask=True):
    normalizers = fit["normalizers"]
    subtask_inputs = _subtask_inputs(rows, fit)
    q = []
    with torch.no_grad():
        for start in range(0, len(rows["a"]), 1024):
            stop = min(start + 1024, len(rows["a"]))
            f = torch.as_tensor(_standardize(rows["f"][start:stop], normalizers["f_mean"], normalizers["f_scale"]), device=device)
            k = torch.as_tensor(_standardize(rows["k"][start:stop], normalizers["k_mean"], normalizers["k_scale"]), device=device)
            s = torch.as_tensor(_standardize(rows["s"][start:stop], normalizers["s_mean"], normalizers["s_scale"]), device=device)
            c = torch.as_tensor(subtask_inputs[start:stop],
                                dtype=torch.long if fit["subtask_mode"] in ("verb_class", "none") else torch.float32,
                                device=device)
            q.append(model.encode_query(
                c, f, k, s, include_img=include_img, include_state=include_state,
                include_subtask=include_subtask).cpu().numpy())
    return np.concatenate(q)


def _subtask_inputs(rows, fit):
    mode = fit["subtask_mode"]
    if mode == "verb_class":
        lookup = fit["class_lookup"]
        if not set(np.unique(rows["skill_class"]).tolist()).issubset(lookup):
            raise ValueError("query class is outside the training classes")
        return np.asarray([lookup[int(x)] for x in rows["skill_class"]], np.int64)
    if mode == "sentence":
        if "subtask_embedding" not in rows:
            raise ValueError("sentence subtask mode requires frozen subtask_embedding rows")
        return np.asarray(rows["subtask_embedding"], np.float32)
    return np.zeros((len(rows["a"]),), np.int64)


def _functional_unit_assignment(unit_builder, values):
    unit_ids = np.full(len(values), -1, np.int64)
    distances = np.full(len(values), np.inf, np.float32)
    for i, value in enumerate(np.asarray(values, np.float32)):
        unit_ids[i], distances[i] = unit_builder._nearest(value)
    return unit_ids, distances


def _unit_scores(gold_b, retrieved_b, random_b, unit_builder, retrieved_unit):
    centers = unit_builder.prototypes
    gold_unit, gold_distance = _functional_unit_assignment(unit_builder, gold_b)
    retrieved_unit = np.asarray(retrieved_unit, np.int64)
    distance = np.linalg.norm(np.asarray(gold_b) - np.asarray(retrieved_b), axis=-1)
    random_distance = np.linalg.norm(np.asarray(gold_b) - np.asarray(random_b), axis=-1)
    covered = gold_distance <= unit_builder.epsilon
    match = (gold_unit == retrieved_unit) & (retrieved_unit >= 0)
    assigned = retrieved_unit >= 0
    return {"mean_b_l2_to_gold": float(np.mean(distance)),
            "median_b_l2_to_gold": float(np.median(distance)),
            "random_training_b_l2_to_gold": float(np.mean(random_distance)),
            "mean_functional_distance_to_nearest_unit": float(np.mean(gold_distance)),
            "unit_match_rate": float(np.mean(match)),
            "strict_covered_unit_match_rate": float(np.mean(match & covered)),
            "gold_nearest_unit_coverage": float(np.mean(covered)),
            "unit_match_given_assigned": float(np.mean(gold_unit[assigned] == retrieved_unit[assigned])) if assigned.any() else None,
            "assigned_unit_fraction": float(np.mean(assigned))}


def _eval_mode(flow, flow_norm, m1, fiii, train, test, train_actions, test_actions,
               train_b, test_b, indices, shuffled_indices, random_indices, retrieved_b, shuffled_b,
               random_b, oracle_b, retrieved_unit, shuffled_unit, unit_builder, *,
               seed, t0, device):
    raw = _eval_actions(flow, flow_norm, m1, fiii, test, indices, shuffled_indices,
                        retrieved_b, shuffled_b, random_b, oracle_b,
                        train_actions[indices], train_actions[shuffled_indices],
                        train_actions[random_indices], test_actions,
                        seed=seed, t0=t0, device=device)
    raw["retrieved_b"] = _unit_scores(test_b, retrieved_b, random_b, unit_builder, retrieved_unit)
    raw["shuffled_b"] = _unit_scores(test_b, shuffled_b, random_b, unit_builder, shuffled_unit)
    return raw


def _suite_horizon(rows, task_indices, horizon):
    suite_rows = _subset(rows, np.isin(rows["task_index"], task_indices))
    return _subset(suite_rows, suite_rows[f"valid_h{horizon}"])


def _fit_m6_units(m1, train, bases, *, seed, device):
    """Run M6's functional DP-means assignment with 32 fixed probes."""
    builder = M6(independence_weight=0.01, kappa=0.1, min_support=10,
                 min_episodes=3, encoder="m1", seed=seed, device=str(device),
                 pending_batch_size=200)
    builder.backbone = m1
    builder.base_dim = int(bases.shape[-1])
    builder.action_shape = tuple(m1.action_shape)
    builder._centers, builder.support_counts, builder.episode_sets = [], [], []
    builder.pending, builder.pending_age = [], {}
    builder.memory = [{"unit_id": None} for _ in range(len(bases))]
    builder.unit_count_history = []
    builder.unit_registration_events = []
    builder.registered_count = builder.merged_count = builder.pending_expired_count = 0
    builder.memory_added_count = builder.memory_rejected_count = 0
    builder._pending_since_cluster = builder._next_pending_id = builder._data_seen = 0
    builder._functional_embedding_cache = {}
    builder._fitted = True
    builder._fit_in_progress = True

    rng = np.random.default_rng(seed + 6046)
    n = len(bases)
    probe_ix = rng.choice(n, size=32, replace=n < 32)
    _, prior_residual = m1.prior(train["f"], train["s"])
    builder._functional_probe = {
        "residual": np.asarray(prior_residual, np.float32)[probe_ix],
        "state": np.asarray(train["s"], np.float32)[probe_ix],
        "feature": np.asarray(train["f"], np.float32)[probe_ix],
    }
    pair_ix = rng.integers(0, n, size=(builder.RANDOM_DISTANCE_PAIRS, 2))
    random_distances = np.asarray([
        builder._functional_distance(bases[left], bases[right])
        for left, right in pair_ix
    ], np.float64)
    builder.functional_median = float(np.median(random_distances))
    builder.epsilon = max(builder.kappa * builder.functional_median, 1e-12)

    episode_ids = np.asarray(train["episode_id"]).astype(str)
    row_unit_ids = np.full(n, -1, np.int64)
    for index in rng.permutation(n):
        builder._data_seen += 1
        unit_id, _, _ = builder._assign(bases[index], episode_ids[index], training=True)
        if unit_id is not None:
            row_unit_ids[index] = unit_id
            builder.memory[index]["unit_id"] = int(unit_id)
        elif builder._last_pending_id is not None:
            for pending in builder.pending:
                if pending["id"] == builder._last_pending_id:
                    pending["memory_index"] = int(index)
                    pending["meta"] = {"training_index": int(index)}
                    break
        builder._cluster_pending()
    builder._cluster_pending(force=True)
    builder._fit_in_progress = False
    for index, item in enumerate(builder.memory):
        if item["unit_id"] is not None:
            row_unit_ids[index] = int(item["unit_id"])
    builder._record_units()
    builder._functional_probe_count = len(probe_ix)
    builder._random_pair_distances = random_distances
    return builder, row_unit_ids


def _run_split_horizon(rows, *, split_name, train_tasks, eval_tasks, horizon,
                       seed, device, flow_epochs, batch_size):
    train = _suite_horizon(rows, train_tasks, horizon)
    test_full = _suite_horizon(rows, eval_tasks, horizon)
    train_classes = np.unique(train["skill_class"])
    known_mask = np.isin(test_full["skill_class"], train_classes)
    test = _subset(test_full, known_mask)
    if not len(train["a"]) or not len(test["a"]):
        raise ValueError(f"empty known-class train/eval data for {split_name} H={horizon}")
    if np.intersect1d(np.unique(train["task_index"]), np.unique(test["task_index"])).size:
        raise AssertionError("train suite and query suite overlap")

    train_actions = train["a"][:, :horizon].copy()
    test_actions = test["a"][:, :horizon].copy()
    flow, flow_norm, flow_fit = train_flow(
        train, horizon=horizon, epochs=flow_epochs, batch_size=batch_size,
        seed=seed + horizon, device=device, patience=8)
    test_h = dict(test)
    test_h["a"] = test_actions
    rng = np.random.default_rng(seed + horizon * 2701)
    eval_noises = [rng.standard_normal((len(test_actions), horizon * test_actions.shape[-1])).astype(np.float32)
                   for _ in range(4)]
    normal_metrics = integrate_flow_sources(flow, flow_norm, test_h, eval_noises,
                                             {"normal": None}, 1.0, device=device)
    normal_mse = float(normal_metrics["normal"])

    m1 = _prepare_m1_train(train, train_actions, seed=seed + horizon,
                           epochs=60, batch_size=512, device=device)
    train_b, _ = m1.encode(train_actions, train["s"])
    test_b, _ = m1.encode(test_actions, test["s"])
    train_mids = _decode_residual_draws(m1, train_b, train["f"], train["s"],
                                        draws=4, seed=seed + horizon * 1000 + 11)
    fiii, fiii_fit = _fit_abstracted_regressor(
        m1, train, train_actions, train_b, train_mids,
        epochs=150, batch_size=batch_size, patience=8,
        seed=seed + horizon * 100 + 3, device=device)

    unit_builder, unit_labels = _fit_m6_units(
        m1, train, train_b, seed=seed + horizon, device=device)
    centers = unit_builder.prototypes.copy()
    if not len(centers):
        raise ValueError("M6 functional unit fit returned no centers")
    center_to_representative = {}
    for unit_id, center in enumerate(centers):
        members = np.flatnonzero(unit_labels == unit_id)
        if len(members):
            center_to_representative[unit_id] = int(members[0])
        else:
            center_to_representative[unit_id] = int(np.linalg.norm(train_b - center[None], axis=1).argmin())
    _, test_functional_distance = _functional_unit_assignment(unit_builder, test_b)
    print(f"M6 units {split_name} H={horizon}: count={len(centers)} "
          f"train_coverage={np.mean(unit_labels >= 0):.3f} "
          f"eval_gold_coverage={np.mean(test_functional_distance <= unit_builder.epsilon):.3f} "
          f"epsilon={unit_builder.epsilon:.6g} probes=32", flush=True)

    key_mean, key_scale = _standardize_fit(_query_key(train))
    class_ids = np.unique(np.concatenate([train["skill_class"], test["skill_class"]]))
    key_train = _query_features(train, _onehot(train["skill_class"], class_ids),
                                "same_class", key_mean, key_scale)
    key_test = _query_features(test, _onehot(test["skill_class"], class_ids),
                               "same_class", key_mean, key_scale)
    _, _, old_indices, _ = retrieve_top1_same_class(
        key_train, train_b, unit_labels, key_test, train["skill_class"], test["skill_class"])

    random_indices = rng.integers(0, len(train_actions), size=len(test_actions))
    result_rows = []
    fit_summaries = []

    def evaluate(method_name, indices, retrieved_b, retrieved_unit, *, distribution=False,
                 fit_tag=None):
        shuffled_indices = _same_class_alternatives(
            train["skill_class"], test["skill_class"], indices,
            seed + horizon * 9001 + len(result_rows) * 13)
        if (shuffled_indices < 0).any():
            raise AssertionError("known-class query has no same-class alternate training interval")
        shuffled_b = train_b[shuffled_indices]
        random_b = train_b[random_indices]
        for t0 in EVAL_T0S:
            values = _eval_mode(
                flow, flow_norm, m1, fiii, train, test, train_actions, test_actions,
                train_b, test_b, indices, shuffled_indices, random_indices,
                retrieved_b, shuffled_b, random_b, test_b,
                retrieved_unit, unit_labels[shuffled_indices], unit_builder,
                seed=seed + horizon * 3301, t0=t0, device=device)
            retrieved_scores = values["retrieved_b"]
            shuffle_scores = values["shuffled_b"]
            result_rows.append({
                "split": split_name, "horizon": horizon, "t0": t0, "method": method_name,
                "query": fit_tag, "rows": len(test_actions),
                "normal_F_action_mse": normal_mse,
                "RAGDP_action_mse": values["ragdp_retrieved"],
                "RAGDP_shuffled_action_mse": values["ragdp_shuffled"],
                "RAGDP_random_action_mse": values["ragdp_random"],
                "RAGDP_oracle_action_mse": values["ragdp_oracle"],
                "iii_action_mse": values["retrieved"],
                "iii_shuffled_action_mse": values["shuffled"],
                "iii_random_action_mse": values["random"],
                "iii_oracle_b_action_mse": values["oracle_b"],
                "iii_minus_RAGDP": values["retrieved"] - values["ragdp_retrieved"],
                "retrieved_b": retrieved_scores,
                "shuffled_b": shuffle_scores,
                "shuffle_ragdp_delta": values["ragdp_shuffled"] - values["ragdp_retrieved"],
                "shuffle_iii_delta": values["shuffled"] - values["retrieved"],
                "distribution_sampled": bool(distribution),
            })

    evaluate("old_key_same_class", old_indices, train_b[old_indices], unit_labels[old_indices],
             fit_tag="legacy key (c)")

    mode_seed = {"none": 0, "verb_class": 1, "sentence": 2}
    for subtask_mode in ("none", "verb_class", "sentence"):
        for nearby_positive in (False, True):
            use_subtask = subtask_mode != "none"
            model, fit = fit_dual_encoder(
                train, train_b, use_subtask=use_subtask,
                subtask_mode=subtask_mode,
                nearby_positive=nearby_positive,
                seed=seed + horizon * 17 + 1000 * mode_seed[subtask_mode] + 2000 * nearby_positive,
                epochs=CLIP_EPOCHS, patience=CLIP_PATIENCE, device=device,
                allowed_task_indices=train_tasks)
            memory, _, h_train = _freeze_memory(model, fit, train, train_b, device=device)
            query = _query_embedding(model, fit, test, device=device)
            top_indices = retrieve_embedding_top1(query, memory.skill_embeddings)
            top_b = memory.bases[top_indices]
            top_units = unit_labels[top_indices]
            tag = f"{subtask_mode} / {'nearby_q05' if nearby_positive else 'instance'}"
            evaluate("clip_top1", top_indices, top_b, top_units, fit_tag=tag)

            # Symmetric retrieval uses the already-trained query tower on both
            # sides: g(query) dot g(stored interval). The g/h training objective
            # and frozen towers remain unchanged; only the memory-side vector
            # changes from h(b) to the stored g embedding.
            symmetric_indices = retrieve_embedding_top1(query, memory.query_embeddings)
            evaluate("clip_symmetric_g_dot_g", symmetric_indices,
                     memory.bases[symmetric_indices], unit_labels[symmetric_indices],
                     fit_tag=tag)

            # Input ablations use the same frozen encoders and stored bases;
            # both sides are re-encoded with the omitted component zeroed.
            # Ablations are reported for the instance-positive fits to avoid
            # duplicating the same diagnostic across the two positive rules.
            if not nearby_positive:
                for omitted in ("image", "state", "subtask"):
                    if omitted == "subtask" and not use_subtask:
                        continue
                    flags = {"include_img": omitted != "image",
                             "include_state": omitted != "state",
                             "include_subtask": omitted != "subtask"}
                    ablated_query = _query_embedding(model, fit, test, device=device, **flags)
                    ablated_memory_query = _query_embedding(
                        model, fit, train, device=device, **flags)
                    ablated_indices = retrieve_embedding_top1(
                        ablated_query, memory.skill_embeddings)
                    ablated_symmetric_indices = retrieve_embedding_top1(
                        ablated_query, ablated_memory_query)
                    evaluate(f"clip_top1_without_{omitted}", ablated_indices,
                             memory.bases[ablated_indices], unit_labels[ablated_indices],
                             fit_tag=tag)
                    evaluate(f"clip_symmetric_without_{omitted}",
                             ablated_symmetric_indices,
                             memory.bases[ablated_symmetric_indices],
                             unit_labels[ablated_symmetric_indices], fit_tag=tag)

            sampled_b, sampled_units, probs = _distribution_unit_sample(
                query, model, fit, centers, seed=seed + horizon * 51 + 7000 * use_subtask
                + 9000 * nearby_positive, device=device)
            rep_indices = np.asarray([center_to_representative[int(u)] for u in sampled_units], np.int64)
            evaluate("clip_unit_distribution", rep_indices, sampled_b, sampled_units,
                     distribution=True, fit_tag=tag)
            fit_summary = {key: value for key, value in fit.items()
                           if key not in ("normalizers", "class_lookup")}
            fit_summary.update({"tower_variant": tag,
                                "query_tower_parameters": sum(p.numel() for p in model.image_tower.parameters())
                                + sum(p.numel() for p in model.state_tower.parameters())
                                + (sum(p.numel() for p in model.subtask_tower.parameters())
                                   + (sum(p.numel() for p in model.class_embedding.parameters())
                                      if model.class_embedding is not None else 0)
                                   if model.subtask_tower is not None else 0)
                                + 2 + (1 if model.w_sub is not None else 0),
                                "modality_weights": model.modality_weights(),
                                "skill_tower_parameters": sum(p.numel() for p in model.skill.parameters()),
                                "distribution_entropy_mean": float(np.mean(-np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1))),
                                "memory_size": int(len(memory.bases))})
            fit_summaries.append(fit_summary)

    return {"split": split_name, "horizon": horizon,
            "train_suite_rows": int(len(train_actions)), "eval_known_class_rows": int(len(test_actions)),
            "train_task_classes": train_classes.tolist(),
            "excluded_unseen_eval_rows": int(len(test_full["a"]) - len(test["a"])),
            "normal_F_action_mse": normal_mse,
            "m1": {"config": m1.describe(), "epochs": m1.epochs_run},
            "Fiii": {"epochs": len(fiii_fit["history"]), "best_epoch": fiii_fit["best_epoch"],
                     "best_validation_loss": fiii_fit["best_validation_loss"]},
            "unit_count": len(centers), "unit_epsilon": unit_builder.epsilon,
            "unit_functional_median_random_base_pairs": unit_builder.functional_median,
            "unit_functional_probe_count": unit_builder._functional_probe_count,
            "unit_random_pair_count": unit_builder.RANDOM_DISTANCE_PAIRS,
            "unit_kappa": unit_builder.kappa, "unit_min_support": unit_builder.min_support,
            "unit_min_episodes": unit_builder.min_episodes,
            "unit_pending_batch_size": unit_builder.pending_batch_size,
            "unit_registered": unit_builder.registered_count,
            "unit_merged": unit_builder.merged_count,
            "unit_pending_after_fit": len(unit_builder.pending),
            "unit_pending_expired": unit_builder.pending_expired_count,
            "unit_training_coverage": float(np.mean(unit_labels >= 0)),
            "unit_eval_gold_coverage": float(np.mean(test_functional_distance <= unit_builder.epsilon)),
            "unit_support_counts": unit_builder.support_counts,
            "unit_episode_counts": [len(episode_set) for episode_set in unit_builder.episode_sets],
            "unit_count_history": unit_builder.unit_count_history,
            "clip_fits": fit_summaries, "results": result_rows}


def _report(summary):
    lines = ["# Learned query embedding retrieval", "",
             "Only training suites enter M1, Fiii, CLIP tower fitting, and memory construction. Split A reports goal rows whose eval_skill_class exists in object+spatial training; unseen classes are excluded. Split B reports spatial rows, all with training-known classes.", "",
             "The query tower separately maps concatenated image/language-free features `(f,k)`, state `s`, and a subtask input to three independently L2-normalized 32D vectors. Subtask modes are none, the existing eval_skill_class verb class, and a short plan phrase selected by eval_skill_segment. Phrase vectors are frozen means of local pi05_base PaliGemma token embeddings; only the E_sub projection is trained. Learned scalar weights scale the vectors before concatenation and final L2 normalization (96D total); the skill tower emits 96D so the InfoNCE spaces match. The table compares ordinary F, legacy `(k,s)` same-class nearest neighbor, asymmetric CLIP retrieval `g(query)·h(b)`, symmetric retrieval `g(query)·g(stored interval)`, and unit-center distribution sampling. Both CLIP retrievals use the same InfoNCE-fitted frozen g and h towers; symmetric retrieval changes only the memory-side embedding. At each t0 in 0.2, 0.4, 0.6, and 0.8 it reports both (iii) and RAGDP action MSE, same-class shuffled values, random and oracle controls, base L2 distance, and M6 functional-unit accuracy. `clip_top1_without_*` and `clip_symmetric_without_*` rows zero that input's embedding on both query and memory sides; ablation rows are run for instance-positive fits.", "",
             "| Split | H | t0 | retrieval | query tower / positive rule | rows | normal F | RAGDP | (iii) | (iii)−RAGDP | RAGDP shuffle | (iii) shuffle | retrieved b L2 | random train b L2 | unit match | gold unit covered | covered match | Δ shuffle iii |",
             "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in summary["splits"]:
        for horizon in split["horizons"]:
            for row in horizon["results"]:
                score = row["retrieved_b"]
                lines.append(f"| {split['name']} | {horizon['horizon']} | {row['t0']:.1f} | {row['method']} | {row['query']} | {row['rows']} | "
                             f"{row['normal_F_action_mse']:.4f} | {row['RAGDP_action_mse']:.4f} | "
                             f"{row['iii_action_mse']:.4f} | {row['iii_minus_RAGDP']:+.4f} | "
                             f"{row['RAGDP_shuffled_action_mse']:.4f} | {row['iii_shuffled_action_mse']:.4f} | "
                             f"{score['mean_b_l2_to_gold']:.4f} | {score['random_training_b_l2_to_gold']:.4f} | "
                             f"{score['unit_match_rate']:.3f} | {score['gold_nearest_unit_coverage']:.3f} | "
                             f"{score['strict_covered_unit_match_rate']:.3f} | {row['shuffle_iii_delta']:+.4f} |")
    lines += ["", "## Best retrieval per t0", "",
              "Best means lowest MSE among memory retrieval rows (legacy key, CLIP top-1, symmetric CLIP, and M6-center distribution); the RAGDP and (iii) winners may differ.", "",
              "| Split | H | t0 | best RAGDP | query variant | RAGDP MSE | best (iii) | query variant | (iii) MSE |",
              "|---|---:|---:|---|---|---:|---|---|---:|"]
    for split in summary["splits"]:
        for horizon in split["horizons"]:
            for t0 in EVAL_T0S:
                rows = [r for r in horizon["results"] if r["t0"] == t0]
                best_ragdp = min(rows, key=lambda r: r["RAGDP_action_mse"])
                best_iii = min(rows, key=lambda r: r["iii_action_mse"])
                lines.append(f"| {split['name']} | {horizon['horizon']} | {t0:.1f} | {best_ragdp['method']} | "
                             f"{best_ragdp['query']} | {best_ragdp['RAGDP_action_mse']:.4f} | "
                             f"{best_iii['method']} | {best_iii['query']} | {best_iii['iii_action_mse']:.4f} |")
    lines += ["", "## Best t0 per retrieval setting", "",
              "| Split | H | retrieval | query variant | best RAGDP t0 | RAGDP MSE | best (iii) t0 | (iii) MSE |",
              "|---|---:|---|---|---:|---:|---:|---:|"]
    for split in summary["splits"]:
        for horizon in split["horizons"]:
            groups = {}
            for row in horizon["results"]:
                groups.setdefault((row["method"], row["query"]), []).append(row)
            for (method, query_variant), rows in groups.items():
                best_ragdp = min(rows, key=lambda r: r["RAGDP_action_mse"])
                best_iii = min(rows, key=lambda r: r["iii_action_mse"])
                lines.append(f"| {split['name']} | {horizon['horizon']} | {method} | {query_variant} | "
                             f"{best_ragdp['t0']:.1f} | {best_ragdp['RAGDP_action_mse']:.4f} | "
                             f"{best_iii['t0']:.1f} | {best_iii['iii_action_mse']:.4f} |")
    lines += ["", "## M6 functional-unit reconstruction", "",
              "Units are rebuilt by skill_bench M6's functional DP-means path, using the same fixed-probe decoded-action metric, top-10 base-L2 prefilter, 64 random base-pair distances for epsilon calibration, seeded row order, pending buffer and support/episode gates. This run uses the requested 32 probes, kappa=0.1, minimum support=10, minimum episodes=3, pending batch=200. Eval coverage means the fraction of held-out gold bases within functional epsilon of a registered unit.", "",
              "| Split | H | units | epsilon | random-pair functional median | train assigned coverage | eval gold coverage | registered | merged | pending at end | support range | episode range |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in summary["splits"]:
        for horizon in split["horizons"]:
            supports = horizon["unit_support_counts"]
            episode_counts = horizon["unit_episode_counts"]
            lines.append(f"| {split['name']} | {horizon['horizon']} | {horizon['unit_count']} | "
                         f"{horizon['unit_epsilon']:.6g} | {horizon['unit_functional_median_random_base_pairs']:.6g} | "
                         f"{horizon['unit_training_coverage']:.3f} | {horizon['unit_eval_gold_coverage']:.3f} | "
                         f"{horizon['unit_registered']} | {horizon['unit_merged']} | {horizon['unit_pending_after_fit']} | "
                         f"{min(supports)}–{max(supports)} | {min(episode_counts)}–{max(episode_counts)} |")
    lines += ["", "## Tower training and modality weights", "",
              "| Split | H | train rows | known-class query rows | excluded unseen-class rows | tower variant | train classes | q05 base distance | temperature | w_img | w_state | w_sub | best epoch | best val loss |",
              "|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in summary["splits"]:
        for horizon in split["horizons"]:
            for fit in horizon["clip_fits"]:
                lines.append(f"| {split['name']} | {horizon['horizon']} | {horizon['train_suite_rows']} | "
                             f"{horizon['eval_known_class_rows']} | {horizon['excluded_unseen_eval_rows']} | {fit['tower_variant']} | "
                             f"{fit['train_task_classes']} | {fit['nearest_cross_episode_b_q05']:.5f} | "
                             f"{fit['temperature']:.4f} | {fit['modality_weights']['image']:.4f} | "
                             f"{fit['modality_weights']['state']:.4f} | {fit['modality_weights']['subtask']:.4f} | "
                             f"{fit['best_epoch']} | {fit['best_validation_loss']:.4f} |")
    lines += ["", "## Interpretation notes", "",
              "InfoNCE uses symmetric in-batch cross entropy with a learned temperature. The nearby-positive model treats other-episode training rows within the training-only q05 threshold as additional positives. The asymmetric method scores g(query) against stored h(b); the symmetric method scores g(query) against stored g(item), with both embeddings computed by the same frozen fitted towers. Learned scalar weights are reported before final query normalization; because the final vector is normalized, their ratios indicate relative modality emphasis. The distribution method samples a unit using softmax(g·h(center)/T); for its RAGDP comparison, the sampled unit's representative training action is used, while (iii) consumes the sampled center directly.", "",
              "Shuffled values are selected from another stored interval with the same eval_skill_class and are evaluated on the same query observations with the same denoising noise. Nearest unit match is the fraction whose retrieved/sampled unit equals the gold base's nearest M6 center, counting unassigned stored values as misses. Gold unit coverage is the fraction within the M6 functional epsilon; covered match requires both that coverage and a correct unit ID. Retrieved b L2 and random-training b L2 are Euclidean distances to the gold M1 base. The per-row `(iii)−RAGDP` column compares both paths using the same retrieval, query rows, t0, and noise seed; negative values favor (iii).", "",
              "The oracle controls use the goal/spatial target action or base and are not memory retrieval results. No environment success metric is measured. H=10 DCT is not part of this study; it is separate from the CLIP query-space comparison.", ""]
    return "\n".join(lines)


def _subtask_comparison_report(summary):
    """Focused mode table with paired same-t0 ablations for sentence inputs."""
    lines = ["# Subtask granularity comparison", "",
             "The table uses instance-only InfoNCE fits for controlled comparisons. Best RAGDP and (iii) t0 are selected independently for each retrieval path. Sentence ablation is evaluated at the corresponding full-sentence best t0, with the same fitted tower, stored memory, evaluation rows, and noise seed; Δ is ablated error minus full-sentence error, so positive values mean removing the subtask text worsens performance.", "",
             "| Split | H | retrieval | subtask mode | best RAGDP (t0) | best (iii) (t0) | b L2 at best (iii) | random b L2 |", "|---|---:|---|---|---:|---:|---:|---:|"]
    ablation_lines = ["", "## Sentence subtask removal at matching t0", "",
                      "| Split | H | retrieval | metric | t0 | sentence | no subtask | Δ no-subtask − sentence | b L2 sentence / no-subtask |",
                      "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for split in summary["splits"]:
        for horizon in split["horizons"]:
            rows = horizon["results"]
            for mode in ("none", "verb_class", "sentence"):
                tag = f"{mode} / instance"
                for method in ("clip_top1", "clip_symmetric_g_dot_g"):
                    selected = [r for r in rows if r["method"] == method and r["query"] == tag]
                    if not selected:
                        continue
                    best_r = min(selected, key=lambda r: r["RAGDP_action_mse"])
                    best_i = min(selected, key=lambda r: r["iii_action_mse"])
                    score = best_i["retrieved_b"]
                    lines.append(f"| {split['name']} | {horizon['horizon']} | {method} | {mode} | "
                                 f"{best_r['RAGDP_action_mse']:.4f} ({best_r['t0']:.1f}) | "
                                 f"{best_i['iii_action_mse']:.4f} ({best_i['t0']:.1f}) | "
                                 f"{score['mean_b_l2_to_gold']:.4f} | "
                                 f"{score['random_training_b_l2_to_gold']:.4f} |")
            for method in ("clip_top1", "clip_symmetric_g_dot_g"):
                full = [r for r in rows if r["method"] == method and r["query"] == "sentence / instance"]
                ablated = [r for r in rows if r["method"] == f"{method.replace('clip_top1', 'clip_top1_without_subtask').replace('clip_symmetric_g_dot_g', 'clip_symmetric_without_subtask')}"
                           and r["query"] == "sentence / instance"]
                if not full or not ablated:
                    continue
                ablation_by_t0 = {r["t0"]: r for r in ablated}
                for metric, field in (("RAGDP", "RAGDP_action_mse"), ("(iii)", "iii_action_mse")):
                    selected = min(full, key=lambda r: r[field])
                    matched = ablation_by_t0[selected["t0"]]
                    ablation_lines.append(
                        f"| {split['name']} | {horizon['horizon']} | {method} | {metric} | "
                        f"{selected['t0']:.1f} | {selected[field]:.4f} | {matched[field]:.4f} | "
                        f"{matched[field] - selected[field]:+.4f} | "
                        f"{selected['retrieved_b']['mean_b_l2_to_gold']:.4f} / "
                        f"{matched['retrieved_b']['mean_b_l2_to_gold']:.4f} |")
    # Add the nearby-positive fit winners separately to show whether that
    # training objective changes the comparison without mixing it into the
    # instance-only controlled ablation above.
    lines.extend(["", "## Nearby-positive sentence fits", "",
                  "| Split | H | retrieval | best RAGDP (t0) | best (iii) (t0) | b L2 at best (iii) | random b L2 |",
                  "|---|---:|---|---:|---:|---:|---:|"])
    for split in summary["splits"]:
        for horizon in split["horizons"]:
            for method in ("clip_top1", "clip_symmetric_g_dot_g"):
                selected = [r for r in horizon["results"] if r["method"] == method
                            and r["query"] == "sentence / nearby_q05"]
                if not selected:
                    continue
                best_r = min(selected, key=lambda r: r["RAGDP_action_mse"])
                best_i = min(selected, key=lambda r: r["iii_action_mse"])
                score = best_i["retrieved_b"]
                lines.append(f"| {split['name']} | {horizon['horizon']} | {method} | "
                             f"{best_r['RAGDP_action_mse']:.4f} ({best_r['t0']:.1f}) | "
                             f"{best_i['iii_action_mse']:.4f} ({best_i['t0']:.1f}) | "
                             f"{score['mean_b_l2_to_gold']:.4f} | {score['random_training_b_l2_to_gold']:.4f} |")
    return "\n".join(lines + ablation_lines)


def run(*, output_root=DEFAULT_OUTPUT_ROOT, cache_dir=DEFAULT_CACHE,
        dataset_dir=DEFAULT_DATASET, seed=0, flow_epochs=150,
        batch_size=512, device_name="cuda", annotations_path=DEFAULT_ANNOTATIONS,
        pi05_base_params=DEFAULT_PI05_BASE_PARAMS,
        paligemma_tokenizer=DEFAULT_PALIGEMMA_TOKENIZER):
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device(device_name)
    rows = load_rows(cache_dir, dataset_dir, max_horizon=max(HORIZONS))
    rows = attach_subtask_texts(rows, cache_dir=cache_dir, annotations_path=annotations_path)
    unique_texts, text_inverse = np.unique(rows["subtask_text"], return_inverse=True)
    unique_embeddings = frozen_pi05_plan_embeddings(
        unique_texts, params_path=pi05_base_params, tokenizer_path=paligemma_tokenizer)
    rows["subtask_embedding"] = unique_embeddings[text_inverse]
    split_summaries = []
    for split_name, (train_tasks, eval_tasks, description) in SPLITS.items():
        horizons = []
        for horizon in HORIZONS:
            result = _run_split_horizon(rows, split_name=split_name,
                                        train_tasks=train_tasks, eval_tasks=eval_tasks,
                                        horizon=horizon, seed=seed, device=device,
                                        flow_epochs=flow_epochs, batch_size=batch_size)
            horizons.append(result)
            print(f"Split {split_name} H={horizon}: units={result['unit_count']}; "
                  f"queries={result['eval_known_class_rows']}; CLIP fits={len(result['clip_fits'])}", flush=True)
        split_summaries.append({"name": split_name, "description": description,
                                "train_task_indices": list(train_tasks),
                                "eval_task_indices": list(eval_tasks), "horizons": horizons})
    output = output_root / (datetime.now(timezone.utc).strftime("intermediate-clip-retrieval-%Y%m%dT%H%M%SZ")
                            + "-" + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    summary = {"seed": seed, "cache": str(cache_dir), "dataset": str(dataset_dir),
               "t0_action": list(EVAL_T0S), "embed_dim": EMBED_DIM,
               "subtask_modes": ["none", "verb_class", "sentence"],
               "plan_annotations": str(annotations_path),
               "frozen_text_embedder": "local pi05_base PaliGemma llm.embedder.input_embedding; sentencepiece token embedding mean; empty segment maps to zero",
               "pi05_base_params": str(pi05_base_params),
               "paligemma_tokenizer": str(paligemma_tokenizer),
               "clip_epochs_max": CLIP_EPOCHS, "clip_batch_size": CLIP_BATCH,
               "clip_patience": CLIP_PATIENCE, "noise_draws": 4,
               "splits": split_summaries}
    (output / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (output / "report.md").write_text(_report(summary))
    (output / "subtask_comparison.md").write_text(_subtask_comparison_report(summary))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flow-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--pi05-base-params", type=Path, default=DEFAULT_PI05_BASE_PARAMS)
    parser.add_argument("--paligemma-tokenizer", type=Path, default=DEFAULT_PALIGEMMA_TOKENIZER)
    args = parser.parse_args()
    out = run(output_root=args.output_root, cache_dir=args.cache_dir,
              dataset_dir=args.dataset, seed=args.seed, flow_epochs=args.flow_epochs,
              batch_size=args.batch_size, device_name=args.device,
              annotations_path=args.annotations, pi05_base_params=args.pi05_base_params,
              paligemma_tokenizer=args.paligemma_tokenizer)
    print(f"output: {out}")

