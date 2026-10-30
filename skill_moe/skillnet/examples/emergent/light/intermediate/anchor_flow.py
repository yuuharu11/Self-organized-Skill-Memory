"""S1: anchor-start flow matching with a frozen, growing M6 skill memory.

All feature vectors are reused from the existing LIBERO-40 cache. Run with
``--smoke`` for the small seed-0 verification, or pass production seeds/splits
without it for the experiment owner's full S1 run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .clip_retrieval import (
    CLIP_EPOCHS, CLIP_PATIENCE, DEFAULT_ANNOTATIONS,
    DEFAULT_PALIGEMMA_TOKENIZER, DEFAULT_PI05_BASE_PARAMS,
    _query_embedding, attach_subtask_texts, fit_dual_encoder,
    frozen_pi05_plan_embeddings,
)
from .splits import SPLITS, subset_rows as _subset
from .experiment import (
    DEFAULT_CACHE, DEFAULT_DATASET, DEFAULT_OUTPUT_ROOT,
    FlowActionHead, _episode_train_validation_indices,
    _standardize, _standardize_fit, load_rows,
)
from ..methods.m6 import M6
from ..methods._common import calibrate

HORIZON = 25
T0_GRID = (0.2, 0.4, 0.6, 0.8, 1.0)
EVAL_SAMPLES = 8
NEW_MOTION_TASKS_A = (12, 15, 16, 19)
GRIP_METRICS = ("positive","negative","initial_command_error")


def parse_source_mix(value):
    """Parse retrieved-anchor,gold-anchor,ordinary-path proportions."""
    values = tuple(float(x.strip()) for x in value.split(","))
    if len(values) != 3 or any(x < 0 for x in values) or sum(values) <= 0:
        raise ValueError("source mix must be three non-negative ratios, e.g. 0.34,0.33,0.33")
    total = sum(values)
    return tuple(x / total for x in values)


def entropy_t0(entropy, unit_count, low, high):
    """Monotone linear confidence map: certain→low t0, diffuse→high t0."""
    if unit_count < 2:
        return np.full_like(np.asarray(entropy, np.float32), float(low))
    normalized = np.clip(np.asarray(entropy, np.float32) / np.log(unit_count), 0.0, 1.0)
    return (float(low) + (float(high) - float(low)) * normalized).astype(np.float32)


def similarity_t0(similarity, reference_min, reference_max, low, high):
    """Map absolute cosine similarity monotonically to noise time t0."""
    span = max(float(reference_max) - float(reference_min), 1e-8)
    confidence = np.clip((np.asarray(similarity, np.float32) - reference_min) / span, 0, 1)
    return (float(high) - (float(high) - float(low)) * confidence).astype(np.float32)


def moment_match_bases(centers, sigmas, weights):
    """Moment-match diagonal Gaussian in b-space, including within-unit spread."""
    centers = np.asarray(centers, np.float32)
    sigmas = np.asarray(sigmas, np.float32)
    weights = np.asarray(weights, np.float32)
    weights = weights / max(float(weights.sum()), 1e-12)
    mean = weights @ centers
    variance = np.sum(weights[:, None] * (
        (centers - mean[None]) ** 2 + sigmas[:, None] ** 2), axis=0)
    return mean.astype(np.float32), np.maximum(variance, 1e-8).astype(np.float32)


def softmax_cosine(query, keys, temperature):
    query = np.asarray(query, np.float32).reshape(-1)
    keys = np.asarray(keys, np.float32)
    if keys.ndim != 2 or keys.shape[1] != query.size or not len(keys):
        raise ValueError("query/key dimensions do not match or key set is empty")
    qn = query / max(float(np.linalg.norm(query)), 1e-8)
    kn = keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8)
    logits = (kn @ qn) / max(float(temperature), 1e-6)
    logits -= logits.max()
    weights = np.exp(logits)
    return (weights / weights.sum()).astype(np.float32)


def state_hash(module):
    digest = hashlib.sha256()
    if hasattr(module, "state_dict"):
        tensors = sorted(module.state_dict().items())
    elif callable(getattr(module, "parameters", None)):
        tensors = [(str(i), value) for i, value in enumerate(module.parameters())]
    else:
        raise TypeError(f"cannot hash model parameters for {type(module).__name__}")
    for name, tensor in tensors:
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class UnitBank:
    """M6 functional units plus query-key means; update is gradient-free."""
    def __init__(self, m6, global_base_scale):
        self.m6 = m6
        self.global_base_scale = max(float(global_base_scale), 1e-6)
        self.members: dict[int, list[tuple[np.ndarray, np.ndarray, str, str]]] = {}
        self.seen_stream_ids: set[int] = set()

    def add(self, unit_id, base, query_key, episode_id, *, stream_id=None):
        if unit_id is None:
            return
        unit_id = int(unit_id)
        if stream_id is not None:
            stream_id = int(stream_id)
            if stream_id in self.seen_stream_ids:
                return
            self.seen_stream_ids.add(stream_id)
        self.members.setdefault(unit_id, []).append((
            np.asarray(base, np.float32).copy(), np.asarray(query_key, np.float32).copy(),
            str(episode_id), str(stream_id if stream_id is not None else "train")))

    def seed_training(self, bases, keys, episodes):
        for base, key, episode in zip(bases, keys, episodes):
            unit_id, distance = self.m6._nearest(base)
            if unit_id is not None and distance <= self.m6.epsilon:
                self.add(unit_id, base, key, episode)

    def consume_registration_events(self, events):
        for event in events:
            meta = event
            key = meta.get("query_key")
            stream_id = meta.get("stream_index")
            if key is None:
                continue
            self.add(event["unit_id"], event["base"], key,
                     meta.get("episode_id", ""), stream_id=stream_id)

    def ids(self):
        return np.asarray(sorted(uid for uid, items in self.members.items() if items), np.int64)

    def centers_keys_sigmas(self, excluded_episodes=()):
        ids = self.ids()
        excluded = set(map(str, excluded_episodes))
        kept_ids, centers, keys, sigmas, episode_sets = [], [], [], [], []
        for uid in ids:
            items = [x for x in self.members[int(uid)] if x[2] not in excluded]
            if not items:
                continue
            bases = np.stack([x[0] for x in items])
            qkeys = np.stack([x[1] for x in items])
            # Leave-recording-out centers must not retain the excluded action
            # through the full unit's prototype or its estimated variance.
            center = (bases.mean(axis=0) if excluded else
                      np.asarray(self.m6.prototypes, np.float32)[uid])
            kept_ids.append(uid); centers.append(center)
            keys.append(qkeys.mean(axis=0))
            # Isotropic standard deviation in b space, before decoding.
            radius = float(np.sqrt(np.mean((bases - center[None]) ** 2)))
            sigmas.append(float(np.clip(radius, 0.03 * self.global_base_scale,
                                        0.75 * self.global_base_scale)))
            episode_sets.append({x[2] for x in items})
        return (np.asarray(kept_ids, np.int64), np.asarray(centers, np.float32),
                np.asarray(keys, np.float32), np.asarray(sigmas, np.float32), episode_sets)

    def posterior(self, query_key, temperature):
        ids, centers, keys, sigmas, episode_sets = self.centers_keys_sigmas()
        if not len(ids):
            raise RuntimeError("no registered M6 units are available for retrieval")
        weights = softmax_cosine(query_key, keys, temperature)
        entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-12))))
        return ids, centers, keys, sigmas, episode_sets, weights, entropy

    def best_excluding_episode(self, query_key, episode_id, temperature):
        ids, centers, keys, sigmas, episode_sets = self.centers_keys_sigmas()
        keep = np.asarray([str(episode_id) not in episodes for episodes in episode_sets])
        if not keep.any():
            raise ValueError("no cross-episode units")
        subset_ids = ids[keep]
        subset_centers = centers[keep]
        subset_keys = keys[keep]
        weights = softmax_cosine(query_key, subset_keys, temperature)
        return int(subset_ids[int(weights.argmax())]), subset_centers[int(weights.argmax())]


def _select_episodes(rows, task_ids, count, seed):
    rng = np.random.default_rng(seed)
    keep = set()
    for task in task_ids:
        task_mask = rows["task_index"] == task
        eps = np.unique(rows["episode_id"][task_mask])
        if count is not None and len(eps) > count:
            eps = np.sort(rng.choice(eps, count, replace=False))
        keep.update(str(x) for x in eps)
    return _subset(rows, np.isin(rows["episode_id"], list(keep)))


def _normal_query_rows(rows):
    result = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in rows.items()}
    result["f"] = np.zeros_like(result["f"], dtype=np.float32)
    return result


def _attach_text_embeddings(rows, params_path, tokenizer_path):
    unique_texts, inverse = np.unique(rows["subtask_text"], return_inverse=True)
    rows["subtask_embedding"] = frozen_pi05_plan_embeddings(
        unique_texts, params_path=params_path, tokenizer_path=tokenizer_path)[inverse]
    return rows


def join_predicted_subtask_rows(rows, jsonl_path, *, allow_missing=False):
    """Exact (episode_id, step) join; optionally retain only covered rows."""
    mapping = {}
    with Path(jsonl_path).open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = (str(item["episode_id"]), int(item["step"]))
            value = str(item["subtask"])
            if key in mapping and mapping[key] != value:
                raise ValueError(f"conflicting predicted subtask for {key} at line {line_number}")
            if key in mapping:
                raise ValueError(f"duplicate predicted subtask key {key} at line {line_number}")
            mapping[key] = value
    joined = []
    keep = []
    missing = []
    for row_index, (episode_id, frame) in enumerate(zip(rows["episode_id"], rows["frame"])):
        key = (str(episode_id), int(frame))
        if key not in mapping:
            missing.append(key)
        else:
            joined.append(mapping[key])
            keep.append(row_index)
    if missing and not allow_missing:
        raise ValueError(f"predicted subtask join missed {len(missing)}/{len(rows['episode_id'])} rows; first={missing[:5]}")
    if len(joined) != len(keep):
        raise AssertionError("predicted subtask join changed row count")
    result = _subset(rows, np.isin(np.arange(len(rows["episode_id"])), keep))
    result["subtask_text"] = np.asarray(joined, dtype=str)
    return result


def predicted_subtask_file(root, split):
    """Resolve only the checkpoint trained without this split's new suite."""
    root = Path(root)
    split_dir = f"split-{split.lower()}"
    # Prefer the originally specified cache-aligned layout, then the producer's
    # current evaluation layout. The latter's run IDs are verified by metrics.
    candidates = [
        root / "cache-aligned" / split_dir / "cache_aligned_subtasks.jsonl",
        root / split_dir / "cache_aligned_subtasks.jsonl",
        root / "evaluation" / split_dir / ("run-2" if split == "A" else "run-1")
        / "cache_aligned_subtasks.jsonl",
    ]
    selected = next((path for path in candidates if path.is_file()), None)
    if selected is None:
        raise FileNotFoundError(f"no predicted subtask file for Split {split} under {root}")
    metrics_path = selected.parent / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text())
        if str(metrics.get("split", "")).upper() != split:
            raise ValueError(f"predicted subtask checkpoint split mismatch: expected {split}, got {metrics.get('split')}")
        expected = "split_a" if split == "A" else "split_b"
        checkpoint = str(metrics.get("checkpoint", "")).lower()
        if expected not in checkpoint:
            raise ValueError(f"predicted subtask checkpoint path does not identify {expected}: {checkpoint}")
    return selected


def attach_subtask_source(rows, source, *, split, cache_dir, annotations_path,
                          predicted_root):
    """Attach gold, empty, whole-instruction, or split-specific predicted text."""
    if source not in ("gold", "none", "instruction", "predicted"):
        raise ValueError(f"unknown subtask source {source!r}")
    if source == "gold":
        return attach_subtask_texts(rows, cache_dir=cache_dir,
                                    annotations_path=annotations_path)
    result = dict(rows)
    if source == "none":
        result["subtask_text"] = np.full(len(rows["episode_id"]), "", dtype=str)
        return result
    manifest = json.loads((Path(cache_dir) / "manifest.json").read_text())
    task_by_episode = {str(item["episode_id"]): str(item["task"])
                       for item in manifest["episodes"]}
    if source == "instruction":
        try:
            result["subtask_text"] = np.asarray(
                [task_by_episode[str(episode)] for episode in rows["episode_id"]], dtype=str)
        except KeyError as exc:
            raise KeyError(f"episode {exc.args[0]} is absent from feature-cache manifest") from exc
        return result
    path = predicted_subtask_file(predicted_root, split)
    return join_predicted_subtask_rows(rows, path)


def _fit_query_encoder(train_rows, bases, *, seed, device, allowed_tasks,
                       subtask_source="gold"):
    qrows = _normal_query_rows(train_rows)
    torch.manual_seed(seed)
    mode = "none" if subtask_source == "none" else "sentence"
    model, fit = fit_dual_encoder(
        qrows, bases, use_subtask=(mode != "none"), subtask_mode=mode,
        nearby_positive=False, seed=seed, epochs=CLIP_EPOCHS,
        patience=CLIP_PATIENCE, device=device, allowed_task_indices=allowed_tasks)
    keys = _query_embedding(model, fit, qrows, device=device)
    return model, fit, keys


def _fit_m6_backbone_and_units(train_rows, *, seed, device, epochs=60):
    """Use M6's exact open-ended fit; M6 owns the fixed M1 backbone."""
    actions = train_rows["a"][:, :HORIZON].copy()
    data = {"actions": actions, "state": train_rows["s"],
            "feature": train_rows["f"], "key": train_rows["k"],
            "episode_id": train_rows["episode_id"],
            "init_state_id": train_rows["episode_id"], "seed": seed}
    m6 = M6(independence_weight=0.01, kappa=0.1, min_support=10,
            min_episodes=3, encoder="m1", width=128, epochs=epochs,
            batch_size=512, seed=seed, device=str(device),
            update_centers_on_add=False)
    m6.fit(data)
    bases, _ = m6.encode(actions, train_rows["s"])
    return m6, bases.astype(np.float32), actions


def _decode_anchor_means(m1, bases, rows, rng, *, batch_size=512):
    output = np.empty((len(bases), *m1.action_shape), np.float32)
    for lo in range(0, len(bases), batch_size):
        hi = min(len(bases), lo + batch_size)
        residual = rng.standard_normal((hi - lo, m1.residual_dim)).astype(np.float32)
        output[lo:hi] = m1.decode(bases[lo:hi], residual,
                                  rows["f"][lo:hi], rows["s"][lo:hi])
    return output


def _anchor_unit_indices(bank, query_keys, episodes, temperature):
    ids, _, keys, _, ep_sets = bank.centers_keys_sigmas()
    normalized = keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8)
    q = np.asarray(query_keys, np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    result = np.zeros(len(q), np.int64)
    for i, (key, episode) in enumerate(zip(q, episodes)):
        eligible = np.asarray([str(episode) not in values for values in ep_sets])
        if not eligible.any():
            raise ValueError("no cross-episode units")
        scores = normalized[eligible] @ key
        result[i] = ids[np.flatnonzero(eligible)[int(scores.argmax())]]
    return result


def _make_flow_sources(m1, bank, train_rows, bases, query_keys, temperature,
                       flow_norm, *, seed):
    rng = np.random.default_rng(seed)
    # Conservatively exclude every recording of the same task during F's
    # retrieved-anchor training: near/duplicate recordings have no source IDs
    # in this cache. Remove their contribution to values, keys AND sigma.
    tasks = train_rows["task_index"]
    views = {}
    for task in np.unique(tasks):
        excluded = train_rows["episode_id"][tasks == task]
        views[int(task)] = bank.centers_keys_sigmas(excluded)
    selected_centers, selected_sigmas = [], []
    for task, key in zip(tasks, query_keys):
        ids, centers, keys, sigmas, _ = views[int(task)]
        if not len(ids):
            raise ValueError("no independent-task anchor candidates; do not relax exclusions")
        pos = int(rng.choice(len(ids), p=softmax_cosine(key,keys,temperature)))
        selected_centers.append(centers[pos]); selected_sigmas.append(sigmas[pos])
    selected_centers = np.asarray(selected_centers,np.float32)
    selected_sigmas = np.asarray(selected_sigmas,np.float32)
    sampled_b = selected_centers + rng.standard_normal(selected_centers.shape).astype(np.float32) * selected_sigmas[:, None]
    memory_mean = _decode_anchor_means(m1, sampled_b, train_rows, rng)
    gold_mean = _decode_anchor_means(m1, bases, train_rows, rng)
    mean, scale = flow_norm["a_mean"], flow_norm["a_scale"]
    memory_norm = _standardize(memory_mean.reshape(len(bases), -1), mean, scale)
    gold_norm = _standardize(gold_mean.reshape(len(bases), -1), mean, scale)
    # Keep calibrated source noise on the standardized action scale.
    sigma_global = float(np.clip(np.median(selected_sigmas), 0.03, 0.75))
    return memory_norm, selected_sigmas, gold_norm, np.full(len(bases), sigma_global, np.float32)


def fit_anchor_flow(train_rows, actions, sources, *, source_mix, seed,
                    epochs, patience, batch_size, device):
    """Fit one F on the ordinary path plus retrieved/gold anchor paths.

    Every path uses x_t=(1-t)*a+t*epsilon. Ordinary paths fit epsilon-A;
    augmented paths fit the clean-action estimate x_t-t*F to A. Using epsilon-A
    off the ordinary path would instead reconstruct A+(1-t)*(a-A).
    The retrieved anchor is the dominant augmented path;
    gold-b anchors remain a small fraction to avoid optimistic retrieval.
    """
    memory_mean, memory_sigma, gold_mean, gold_sigma = sources
    train_ix, valid_ix = _episode_train_validation_indices(train_rows, seed + 11)
    f_mean, f_scale = _standardize_fit(train_rows["f"][train_ix])
    s_mean, s_scale = _standardize_fit(train_rows["s"][train_ix])
    a_mean, a_scale = _standardize_fit(actions[train_ix].reshape(len(train_ix), -1))
    dim = int(np.prod(actions.shape[1:]))
    at = torch.as_tensor(_standardize(actions.reshape(len(actions), dim), a_mean, a_scale), device=device)
    ft = torch.as_tensor(_standardize(train_rows["f"], f_mean, f_scale), device=device)
    st = torch.as_tensor(_standardize(train_rows["s"], s_mean, s_scale), device=device)
    mem = torch.as_tensor(memory_mean, device=device)
    gold = torch.as_tensor(gold_mean, device=device)
    torch.manual_seed(seed)
    model = FlowActionHead(ft.shape[-1], st.shape[-1], dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    generator = torch.Generator(device=device).manual_seed(seed + 29)
    source_probs = torch.as_tensor(source_mix, device=device)
    val_rng = np.random.default_rng(seed + 31)
    val_sources = val_rng.choice(3, size=len(valid_ix), p=source_mix)
    val_eps = torch.as_tensor(val_rng.standard_normal((len(valid_ix), dim)).astype(np.float32), device=device)
    val_anchor = at[valid_ix].clone()
    val_anchor[val_sources == 0] = mem[valid_ix[val_sources == 0]]
    val_anchor[val_sources == 1] = gold[valid_ix[val_sources == 1]]
    val_t = torch.rand((len(valid_ix),), generator=torch.Generator(device=device).manual_seed(seed + 32), device=device)
    val_xt = (1 - val_t[:, None]) * val_anchor + val_t[:, None] * val_eps
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(epochs):
        order = torch.as_tensor(train_ix, device=device)
        order = order[torch.randperm(len(order), generator=generator, device=device)]
        losses = []
        model.train()
        for start in range(0, len(order), batch_size):
            ix = order[start:start + batch_size]
            kind = torch.multinomial(source_probs.expand(len(ix), -1), 1,
                                     generator=generator).squeeze(-1)
            eps = torch.randn((len(ix), dim), generator=generator, device=device)
            anchor = at[ix].clone()
            mi = kind == 0
            gi = kind == 1
            anchor[mi] = mem[ix[mi]]
            anchor[gi] = gold[ix[gi]]
            t = torch.rand((len(ix),), generator=generator, device=device)
            xt = (1 - t[:, None]) * anchor + t[:, None] * eps
            pred = model(xt, t, ft[ix], st[ix])
            loss = _flow_loss(pred,xt,t,eps,at[ix],kind)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vpred = model(val_xt, val_t, ft[valid_ix], st[valid_ix])
            val_loss = float(_flow_loss(vpred,val_xt,val_t,val_eps,at[valid_ix],
                                        torch.as_tensor(val_sources,device=device)).cpu())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "validation_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"f_mean": f_mean, "f_scale": f_scale, "s_mean": s_mean,
                   "s_scale": s_scale, "a_mean": a_mean, "a_scale": a_scale,
                   "action_shape": tuple(actions.shape[1:])}, {
        "history": history, "best_epoch": best_epoch,
        "best_validation_loss": best_loss, "source_mix": list(source_mix),
        "train_rows": len(train_ix), "validation_rows": len(valid_ix)}


def _flow_loss(velocity, xt, t, eps, target, source_kind):
    ordinary = (velocity-(eps-target)).square().mean(dim=-1)
    anchored = (xt-t[:,None]*velocity-target).square().mean(dim=-1)
    return torch.where(source_kind == 2,ordinary,anchored).mean()


def _rollout_flow(model, norm, rows, sources, noises, t0_values, *, steps=10,
                  device, chunk=128):
    """K-step Euler flow rollout; sources None means start at pure noise."""
    n = len(rows["a"])
    dim = int(np.prod(norm["action_shape"]))
    noises = np.asarray(noises, np.float32).reshape(-1, n, dim)
    samples = noises.shape[0]
    t0 = np.broadcast_to(np.asarray(t0_values, np.float32), (n,)).copy()
    source_arr = None if sources is None else np.asarray(sources, np.float32).reshape(samples, n, dim)
    predictions = np.empty((samples, n, dim), np.float32)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        count = hi - lo
        eps = torch.as_tensor(noises[:, lo:hi].reshape(samples * count, dim), device=device)
        t0_t = torch.as_tensor(t0[lo:hi], device=device).repeat(samples)
        if source_arr is None:
            x = eps.clone()
        else:
            src = torch.tensor(source_arr[:, lo:hi].reshape(samples * count, dim), device=device)
            x = t0_t[:, None] * eps + (1 - t0_t[:, None]) * src
        f = torch.as_tensor(_standardize(rows["f"][lo:hi], norm["f_mean"], norm["f_scale"]), device=device).repeat(samples, 1)
        s = torch.as_tensor(_standardize(rows["s"][lo:hi], norm["s_mean"], norm["s_scale"]), device=device).repeat(samples, 1)
        with torch.no_grad():
            for step in range(steps):
                t_now = t0_t * (1 - step / steps)
                dt = -t0_t / steps
                x = x + dt[:, None] * model(x, t_now, f, s)
        predictions[:, lo:hi] = (x.reshape(samples, count, dim).cpu().numpy()
                                  * norm["a_scale"] + norm["a_mean"])
    return predictions.reshape(samples, n, *norm["action_shape"])


def _draw_decoded_samples(m1, bases, sigmas, rows, norm, rng, *, draws=EVAL_SAMPLES):
    """Draw b'~N(c,sigma^2 I) in base space, then decode Dec(b',r,s)."""
    bases = np.asarray(bases, np.float32)
    if bases.ndim == 2:
        bases = np.broadcast_to(bases[None], (draws, *bases.shape)).copy()
    n = bases.shape[1]
    sigmas = np.asarray(sigmas, np.float32)
    if sigmas.ndim == 1:
        sigmas = np.broadcast_to(sigmas[None], (draws, n))
    if sigmas.ndim == 2:
        sigmas = sigmas[..., None]
    sampled_b = bases + rng.standard_normal(bases.shape).astype(np.float32) * sigmas
    flat_b = sampled_b.reshape(draws * n, -1)
    flat_s = np.tile(rows["s"], (draws, 1))
    flat_f = np.tile(rows["f"], (draws, 1))
    residual = rng.standard_normal((draws * n, m1.residual_dim)).astype(np.float32)
    decoded = m1.decode(flat_b, residual, flat_f, flat_s).reshape(draws, n, -1)
    return _standardize(decoded, norm["a_mean"], norm["a_scale"])


def _neighbor_blend(ids, centers, weights, sigmas, m6):
    top = int(np.argmax(weights))
    top_id = int(ids[top])
    candidates = np.argsort(weights)[::-1][:min(5, len(ids))]
    candidate_centers = centers[candidates]
    d = m6._functional_distances(centers[top], candidate_centers)
    near = candidates[np.asarray(d) <= m6.epsilon]
    if not len(near):
        near = np.asarray([top])
    near_w = weights[near]
    near_w = near_w / near_w.sum()
    mean, variance = moment_match_bases(centers[near], sigmas[near], near_w)
    return mean, float(np.sqrt(np.mean(variance))), top_id


def _gripper_timing_tv(prediction, target, *, threshold=0.1):
    """TV of first sign transitions; initial held command is not an event."""
    h = prediction.shape[-2]
    def hist(actions, positive):
        values = actions[..., 6]
        sign = np.where(values > threshold,1,np.where(values < -threshold,-1,0))
        previous = np.zeros(sign.shape[:-1],np.int8)
        active = np.zeros_like(sign,dtype=bool)
        destination = 1 if positive else -1
        for step in range(h):
            current = sign[...,step]
            active[...,step] = (current == destination) & (previous == -destination)
            previous = np.where(current != 0,current,previous)
        any_active = active.any(axis=-1)
        first = active.argmax(axis=-1)
        event = np.where(any_active, first, h)
        counts = np.bincount(event.reshape(-1), minlength=h + 1).astype(np.float64)
        return counts / max(counts.sum(), 1.0)
    output = {}
    for name, positive in (("positive", True), ("negative", False)):
        p, q = hist(prediction, positive), hist(target[None], positive)
        output[name] = float(0.5 * np.abs(p - q).sum())
    return output


def _sample_mse(prediction, target):
    error = np.mean((prediction - target[None]) ** 2, axis=tuple(range(2, prediction.ndim)))
    return {"mse": float(error.mean()), "best_of_8_mse": float(error.min(axis=0).mean())}


def _fit_t0_calibration(flow, flow_norm, m1, fiii, bank, model_q, fit,
                        val_rows, val_b, *, raw_memory, seed, device,
                        calibration_limit=256):
    if calibration_limit and len(val_rows["a"]) > calibration_limit:
        choose = np.sort(np.random.default_rng(seed).choice(len(val_rows["a"]), calibration_limit, replace=False))
        val_rows = _subset(val_rows, np.isin(np.arange(len(val_rows["a"])), choose))
        val_b = val_b[choose]
    qrows = _normal_query_rows(val_rows)
    q = _query_embedding(model_q, fit, qrows, device=device)
    ids, centers, keys, sigmas, _ = bank.centers_keys_sigmas()
    weights = np.stack([softmax_cosine(x, keys, fit["temperature"]) for x in q])
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    kn = keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8)
    max_similarity = np.max(qn @ kn.T, axis=1)
    rng = np.random.default_rng(seed + 17)
    selected = np.stack([rng.choice(len(ids),size=EVAL_SAMPLES,p=w) for w in weights],axis=1)
    decoded_norm = _draw_decoded_samples(m1, centers[selected], sigmas[selected], val_rows,
                                         flow_norm, rng, draws=EVAL_SAMPLES)
    epsilon = rng.standard_normal(decoded_norm.shape).astype(np.float32)
    true_action = qrows["a"][:, :HORIZON]
    calib = {"fixed": {}, "entropy": {}, "absolute_similarity": {}}
    for t0 in T0_GRID:
        pred = _rollout_flow(flow, flow_norm, val_rows, decoded_norm, epsilon,
                             np.full(len(q), t0, np.float32), steps=10, device=device)
        calib["fixed"][str(t0)] = float(np.mean((pred - true_action[None]) ** 2))
    fixed = float(min(T0_GRID, key=lambda x: calib["fixed"][str(x)]))
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1)
    for low in T0_GRID:
        for high in T0_GRID:
            if high < low:
                continue
            values = entropy_t0(entropy, len(ids), low, high)
            pred = _rollout_flow(flow, flow_norm, val_rows, decoded_norm, epsilon,
                                 values, steps=10, device=device)
            score = float(np.mean((pred - true_action[None]) ** 2))
            calib["entropy"][f"{low},{high}"] = score
            values = similarity_t0(max_similarity, float(max_similarity.min()),
                                   float(max_similarity.max()), low, high)
            pred = _rollout_flow(flow, flow_norm, val_rows, decoded_norm, epsilon,
                                 values, steps=10, device=device)
            calib["absolute_similarity"][f"{low},{high}"] = float(np.mean((pred - true_action[None]) ** 2))
    pair = min((tuple(map(float, k.split(","))) for k in calib["entropy"]),
               key=lambda p: calib["entropy"][f"{p[0]},{p[1]}"])
    abs_pair = min((tuple(map(float, k.split(","))) for k in calib["absolute_similarity"]),
                   key=lambda p: calib["absolute_similarity"][f"{p[0]},{p[1]}"])

    raw_keys, raw_actions, raw_episodes = raw_memory
    raw_sources, raw_similarity = [], []
    for qi, episode in zip(q, qrows["episode_id"]):
        action, similarity = _raw_top1(qi, raw_keys, raw_actions,
                                       raw_episodes, episode, None)
        raw_sources.append(action)
        raw_similarity.append(similarity)
    raw_sources = _standardize(np.asarray(raw_sources).reshape(len(q), -1),
                               flow_norm["a_mean"], flow_norm["a_scale"])
    raw_sources = np.broadcast_to(raw_sources[None],(EVAL_SAMPLES,*raw_sources.shape))
    raw_similarity = np.asarray(raw_similarity, np.float32)
    rts_scores, ragdp_scores = {}, {}
    for t0 in T0_GRID:
        pred = _rollout_flow(flow,flow_norm,val_rows,raw_sources,epsilon,
                             np.full(len(q),t0,np.float32),steps=10,device=device)
        ragdp_scores[str(t0)] = float(np.mean((pred-true_action[None])**2))
    for low in T0_GRID:
        for high in T0_GRID:
            if high < low:
                continue
            values = similarity_t0(raw_similarity, float(raw_similarity.min()),
                                   float(raw_similarity.max()), low, high)
            pred = _rollout_flow(flow, flow_norm, val_rows, raw_sources, epsilon,
                                 values, steps=10, device=device)
            rts_scores[f"{low},{high}"] = float(np.mean((pred - true_action[None]) ** 2))
    rts_pair = min((tuple(map(float, k.split(","))) for k in rts_scores),
                   key=lambda p: rts_scores[f"{p[0]},{p[1]}"])
    calib.update({"selected_fixed_t0": fixed,
                  "selected_ragdp_fixed_t0":float(min(T0_GRID,key=lambda x:ragdp_scores[str(x)])),
                  "ragdp_fixed_scores":ragdp_scores,
                  "selected_entropy_low_high": list(pair),
                  "selected_absolute_similarity_low_high": list(abs_pair),
                  "selected_rts_similarity_low_high": list(rts_pair),
                  "absolute_similarity_reference": [float(max_similarity.min()), float(max_similarity.max())],
                  "rts_similarity_reference": [float(raw_similarity.min()), float(raw_similarity.max())],
                  "validation_rows": len(q),
                  "best_fixed_validation_mse": calib["fixed"][str(fixed)],
                  "best_entropy_validation_mse": calib["entropy"][f"{pair[0]},{pair[1]}"],
                  "best_absolute_similarity_validation_mse": calib["absolute_similarity"][f"{abs_pair[0]},{abs_pair[1]}"],
                  "best_rts_validation_mse": rts_scores[f"{rts_pair[0]},{rts_pair[1]}"]})
    return calib


def _prepare_split_rows(rows, split_name, *, smoke, seed, smoke_train_episodes,
                        smoke_eval_episodes):
    train_tasks, eval_tasks, _ = SPLITS[split_name]
    train = _subset(rows, np.isin(rows["task_index"], train_tasks))
    test_full = _subset(rows, np.isin(rows["task_index"], eval_tasks))
    train = _subset(train, train["valid_h25"])
    test_full = _subset(test_full, test_full["valid_h25"])
    # Sentence queries support unseen verbs. Removing them would remove the
    # very novel motions required by S1, and make S1-b unmeasurable.
    test = test_full
    excluded = 0
    if smoke:
        train = _select_episodes(train, train_tasks, smoke_train_episodes, seed + 101)
        test = _select_episodes(test, eval_tasks, smoke_eval_episodes, seed + 211)
    return train, test, int(excluded)


def _m6_new_suite_stream(m6, bank, eval_rows, eval_b, eval_q, *, temperature,
                         initial_unit_ids, train_keys, train_actions,
                         train_episodes, seed):
    """Replay new-suite rows in deterministic chunks: retrieve, then add."""
    rng = np.random.default_rng(seed)
    # Retrieve every interval in an episode BEFORE inserting any of its values.
    # Window shuffling otherwise makes neighboring/future windows searchable.
    episodes = np.asarray(eval_rows["episode_id"]).astype(str)
    episode_order = rng.permutation(np.unique(episodes))
    blocks = []
    for episode in episode_order:
        ix = np.flatnonzero(episodes == episode)
        if "frame" in eval_rows:
            ix = ix[np.argsort(eval_rows["frame"][ix],kind="stable")]
        blocks.append(ix)
    order = np.concatenate(blocks)
    initial_unit_ids = set(int(x) for x in initial_unit_ids)
    output = []
    interval_store = []
    for rows_ix in blocks:
        ids, centers, keys, sigmas, _ = bank.centers_keys_sigmas()
        # Evaluate each row against memory as it stood at the beginning of the
        # chunk. This is causal at chunk granularity; each row is added after.
        for j, old in enumerate(rows_ix):
            query = eval_q[old]
            weights = softmax_cosine(query,keys,temperature)
            entropy = float(-np.sum(weights*np.log(np.maximum(weights,1e-12))))
            best_pos = int(weights.argmax())
            best_id = int(ids[best_pos])
            best_center = centers[best_pos]
            top_before = best_id in initial_unit_ids
            # Raw action retrieval uses fit-suite memory plus earlier new-suite
            # rows from other episodes, matching M6's episode exclusion.
            raw_action, raw_similarity = _raw_top1(
                query, train_keys, train_actions, train_episodes,
                eval_rows["episode_id"][old], interval_store)
            normalized_query = query / max(float(np.linalg.norm(query)), 1e-8)
            normalized_keys = keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8)
            max_similarity = float(np.max(normalized_keys @ normalized_query))
            output.append({"row_index": int(old), "unit_ids": ids, "centers": centers,
                           "unit_keys": keys, "weights": weights, "sigmas": sigmas,
                           "best_id": best_id, "best_center": best_center,
                           "entropy": entropy, "max_similarity": max_similarity,
                           "raw_similarity": raw_similarity,
                           "selected_post_add": (best_id not in initial_unit_ids),
                           "raw_action": raw_action})
        before_event = len(m6.unit_registration_events)
        for old in rows_ix:
            meta = {"episode_id": str(eval_rows["episode_id"][old]),
                    "init_state_id": str(eval_rows["episode_id"][old]),
                    "stream_index": int(old), "query_key": eval_q[old].copy()}
            m6.memory_add(eval_q[old], eval_b[old], meta)
            interval_store.append({"episode_id": str(eval_rows["episode_id"][old]),
                                   "key": eval_q[old].copy(),
                                   "base": eval_b[old].copy(),
                                   "action": eval_rows["a"][old, :HORIZON].copy()})
        # Flush eligible support at episode boundaries, not only after 200
        # novel windows. Age pending rows by actual new evidence, rather than
        # charging a whole 200-window batch for a short episode.
        pending = int(m6._pending_since_cluster)
        if pending:
            original_batch_size = m6.pending_batch_size
            try:
                m6.pending_batch_size = pending
                m6._cluster_pending(force=True)
            finally:
                m6.pending_batch_size = original_batch_size
        new_events = m6.unit_registration_events[before_event:]
        bank.consume_registration_events(new_events)
        # Immediate assignments do not produce registration events.
        for old in rows_ix:
            stream_id = int(old)
            if stream_id in bank.seen_stream_ids:
                continue
            uid, distance = m6._nearest(eval_b[old])
            if uid is not None and distance <= m6.epsilon:
                bank.add(uid, eval_b[old], eval_q[old],
                         eval_rows["episode_id"][old], stream_id=stream_id)
    return output, order


def _raw_top1(query_key, train_keys, train_actions, train_episodes, query_episode,
              prior_new):
    keys = [train_keys]
    acts = [train_actions]
    eps = [np.asarray(train_episodes).astype(str)]
    if prior_new:
        keys.append(np.stack([x["key"] for x in prior_new]))
        acts.append(np.stack([x["action"] for x in prior_new]))
        eps.append(np.asarray([x["episode_id"] for x in prior_new]).astype(str))
    key_matrix, action_matrix, episode_vec = np.concatenate(keys), np.concatenate(acts), np.concatenate(eps)
    mask = episode_vec != str(query_episode)
    if not mask.any():
        raise ValueError("no cross-episode raw-memory candidates")
    q = np.array(query_key, np.float32, copy=True)
    q /= max(float(np.linalg.norm(q)), 1e-8)
    kn = key_matrix[mask]
    kn /= np.maximum(np.linalg.norm(kn, axis=1, keepdims=True), 1e-8)
    similarities = kn @ q
    local = int(similarities.argmax())
    selected = np.flatnonzero(mask)[local]
    return action_matrix[selected], float(similarities[local])


def _metrics(pred, target):
    score = _sample_mse(pred, target)
    score["gripper_timing_tv"] = _gripper_timing_tv(pred, target)
    def initial(a):
        values = a[...,6]
        active = np.abs(values) > .1
        ix = active.argmax(axis=-1)
        value = np.take_along_axis(values,ix[...,None],axis=-1)[...,0]
        return np.where(active.any(axis=-1),np.sign(value),0)
    score["gripper_timing_tv"]["initial_command_error"] = float(np.mean(initial(pred)!=initial(target)[None]))
    return score


def _post_add_rate(flags,selected):
    flags = np.asarray(flags,dtype=bool)
    return float(np.mean(np.asarray(selected,dtype=bool)[flags])) if flags.any() else None


def _novel_motion_mask(rows,split,training_classes):
    # The four novel tasks also contain ordinary pick/place intervals. Those
    # intervals must not inflate/dilute S1-b's novel-motion denominator.
    return ((split == 'A') & np.isin(rows['task_index'],NEW_MOTION_TASKS_A)
            & ~np.isin(rows['skill_class'],training_classes))


def _report(summary):
    source_desc = {
        "gold": "gold segment annotation (upper bound)",
        "none": "no subtask input (language-free image and state)",
        "instruction": "whole task instruction, constant within episode",
        "predicted": "split-specific pi05 predicted subtask joined by (episode_id, step)",
    }[summary["subtask_source"]]
    lines = [
        "# Anchor-flow S1 results",
        "",
        f"M1 and E were fitted on Split A/B training suites only and frozen. Query subtask source: {source_desc}. E always uses language-free k and proprioception (f zeroed only in E). All valid H=25 training-suite rows are retained in each source condition; predicted mode requires exact (episode_id, step) coverage for training and query rows. One shared F and 10 Euler steps are used by ALL comparisons, including iii_top1. F fits epsilon-A on ordinary paths and the clean-action estimate x_t-t*F to A on augmented anchor paths. Retrieved-anchor training leaves out all recordings of the query task, including their contributions to centers, keys and sigma; no exclusion fallback is allowed. This conservative rule covers nearby recordings within a task because source/init IDs are absent.",
        "",
        "The stream shuffles whole episodes: every window is evaluated before any value from that episode is inserted. M6 is flushed at episode boundaries without lowering support=10 / episodes=3 / epsilon; pending age advances by actual new evidence. No unknown motion classes are removed. Component samples and isotropic sigma are in b space before decoding. OptimusVLA moment matching is also in b space. Fixed and confidence t0 (including t0=1 fallback) are calibrated on independent training-suite episodes using the same sampled-component distribution and real F features as evaluation. RAGDP has its own validation-selected fixed t0. White noise is common across all methods, and latent/residual draws are shared across anchor ablations.",
        "",
        "Best-of-8 is the mean of each interval's minimum MSE among eight draws. Timing TV compares first positive/negative sign SWITCHES (threshold ±0.1, zero holds the prior sign; an initial held sign is not a switch; no switch is a separate bin). Initial-command error is also reported so constant opposite commands cannot falsely pass. S1-b divides only by new-motion queries; it is causal prequential retrieval, including the warm-up needed to register a unit. Small smoke data do not decide any S1 gate. Initial-state separation is not established; no inference about online success follows from offline MSE.",
        "",
        "| Split | Seed | H | Population | Condition | N | Action MSE | Best-of-8 MSE | Positive timing TV | Negative timing TV | Initial-command error | New-motion top-1 post-add unit |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        def fmt(x): return "—" if x is None else f"{x:.4f}"
        grip = row["gripper_timing_tv"]
        lines.append(f"| {row['split']} | {row['seed']} | {row['horizon']} | {row['population']} | {row['condition']} | {row['n']} | {fmt(row['mse'])} | {fmt(row['best_of_8_mse'])} | {fmt(grip['positive'])} | {fmt(grip['negative'])} | {fmt(grip['initial_command_error'])} | {fmt(row.get('new_motion_post_add_top1'))} |")
    lines.extend(["", "## S1 gates", ""])
    for fit in summary["runs"]:
        lines.append(f"- Split {fit['split']} seed {fit['seed']}: S1-a mixture improvement vs normal={fit['s1a_relative_improvement']:.1%}; vs RAGDP={fit['s1a_vs_ragdp']:.1%}. S1-b new-motion post-add top-1={fmt(fit['s1b_post_add_top1'])}. Fixed t0={fit['t0_calibration']['selected_fixed_t0']:.1f}; entropy map low/high={fit['t0_calibration']['selected_entropy_low_high']}.")
        for population,gates in fit.get('gate_point_estimates',{}).items():
            lines.append(f"  - {population}: {json.dumps(gates)}. All S1-a clauses and S1-c metrics must be checked; improvement vs normal alone is insufficient. These are point estimates, not a statistical verdict.")
    lines.extend(["", "## Reproduction", "",
                  f"Smoke: `cd /work && source env/activate.sh && XLA_PYTHON_CLIENT_PREALLOCATE=false python -m skill_moe.skillnet.examples.emergent.light.intermediate.anchor_flow --smoke --seeds 0 --splits A,B --subtask-source {summary['subtask_source']} --device cuda`.",
                  "Full S1 (run once per source: gold, none, instruction, predicted): `cd /work && source env/activate.sh && XLA_PYTHON_CLIENT_PREALLOCATE=false python -m skill_moe.skillnet.examples.emergent.light.intermediate.anchor_flow --seeds 0,1,2 --splits A,B --subtask-source SOURCE --device cuda`.",
                  f"Observed smoke duration: {summary.get('smoke_duration_seconds', 0):.0f} s. Full-run wall-time estimate: {summary.get('full_run_estimate', 'pending')}.", ""])
    return "\n".join(lines)


def _run_one(rows, *, split, seed, device, source_mix, smoke,
             smoke_train_episodes, smoke_eval_episodes,
             m1_epochs, f_epochs, fiii_epochs, batch_size,
             subtask_source, cache_dir, predicted_root):
    tic = time.perf_counter()
    train, test, excluded = _prepare_split_rows(
        rows, split, smoke=False, seed=seed,
        smoke_train_episodes=smoke_train_episodes,
        smoke_eval_episodes=smoke_eval_episodes)
    original_train_rows = len(train["episode_id"])
    if subtask_source == "predicted":
        prediction_path = predicted_subtask_file(predicted_root, split)
        train = join_predicted_subtask_rows(train, prediction_path)
        test = join_predicted_subtask_rows(test, prediction_path)
    prediction_covered_train_rows = len(train["episode_id"])
    prediction_missing_train_rows = original_train_rows - prediction_covered_train_rows
    if smoke:
        train_tasks, eval_tasks, _ = SPLITS[split]
        train = _select_episodes(train, train_tasks, smoke_train_episodes, seed + 101)
        test = _select_episodes(test, eval_tasks, smoke_eval_episodes, seed + 211)
    train = attach_subtask_source(
        train, subtask_source, split=split, cache_dir=cache_dir,
        annotations_path=DEFAULT_ANNOTATIONS, predicted_root=predicted_root)
    test = attach_subtask_source(
        test, subtask_source, split=split, cache_dir=cache_dir,
        annotations_path=DEFAULT_ANNOTATIONS, predicted_root=predicted_root)
    if set(train["episode_id"]) & set(test["episode_id"]):
        raise ValueError("training/evaluation episodes overlap")
    # A separate episode-level validation stream calibrates fixed and adaptive t0.
    fit_ix, val_ix = _episode_train_validation_indices(train, seed + 401)
    train_fit = _subset(train, np.isin(np.arange(len(train["a"])), fit_ix))
    validation = _subset(train, np.isin(np.arange(len(train["a"])), val_ix))
    if subtask_source != "none":
        train_fit = _attach_text_embeddings(train_fit, DEFAULT_PI05_BASE_PARAMS,
                                            DEFAULT_PALIGEMMA_TOKENIZER)
        validation = _attach_text_embeddings(validation, DEFAULT_PI05_BASE_PARAMS,
                                             DEFAULT_PALIGEMMA_TOKENIZER)
        test = _attach_text_embeddings(test, DEFAULT_PI05_BASE_PARAMS,
                                       DEFAULT_PALIGEMMA_TOKENIZER)
    train_actions = train_fit["a"][:, :HORIZON].copy()
    m6, bases, _ = _fit_m6_backbone_and_units(
        train_fit, seed=seed + 19, device=device, epochs=m1_epochs)
    m1 = m6.backbone
    base_before = state_hash(m1)
    query_model, query_fit, train_keys = _fit_query_encoder(
        train_fit, bases, seed=seed + 701, device=device,
        allowed_tasks=tuple(SPLITS[split][0]), subtask_source=subtask_source)
    # M6's key threshold must be calibrated in the same E-space it will receive
    # when new-suite observations are inserted.
    m6._key_tau, m6._key_d_ref = calibrate(
        train_keys, train_fit["episode_id"], train_fit["episode_id"], quantile=0.95)
    bank = UnitBank(m6, float(np.sqrt(np.mean((bases - bases.mean(axis=0)) ** 2))))
    bank.seed_training(bases, train_keys, train_fit["episode_id"])
    if not len(bank.ids()):
        raise RuntimeError("M6 produced no assignable training units")
    # M1 and E never change after this point; only M6's non-gradient memory does.
    m1_hash, e_hash = state_hash(m1), state_hash(query_model)

    val_b, _ = m1.encode(validation["a"][:, :HORIZON], validation["s"])
    # (iii) is the top-1 start ablation of the SAME F, not a separately fitted
    # direct-regression head with a different training budget.
    fiii = None
    # Preliminary normalizer for anchor generation; F's own fit below returns
    # the final one. It is computed only from the training subset.
    norm_train_ix, _ = _episode_train_validation_indices(train_fit, seed + 827 + 11)
    flow_a_mean, flow_a_scale = _standardize_fit(
        train_actions[norm_train_ix].reshape(len(norm_train_ix), -1))
    flow_f_mean, flow_f_scale = _standardize_fit(train_fit["f"][norm_train_ix])
    flow_s_mean, flow_s_scale = _standardize_fit(train_fit["s"][norm_train_ix])
    flow_norm_seed = {"a_mean": flow_a_mean, "a_scale": flow_a_scale,
                      "f_mean": flow_f_mean, "f_scale": flow_f_scale,
                      "s_mean": flow_s_mean, "s_scale": flow_s_scale,
                      "action_shape": m1.action_shape}
    source_train = _make_flow_sources(
        m1, bank, train_fit, bases, train_keys, query_fit["temperature"],
        flow_norm_seed, seed=seed + 823)
    flow, flow_norm, flow_fit = fit_anchor_flow(
        train_fit, train_actions, source_train, source_mix=source_mix,
        seed=seed + 827, epochs=f_epochs, patience=8,
        batch_size=batch_size, device=device)
    flow_hash = state_hash(flow)

    val_qrows = _normal_query_rows(validation)
    val_keys = _query_embedding(query_model, query_fit, val_qrows, device=device)
    val_b, _ = m1.encode(validation["a"][:, :HORIZON], validation["s"])
    t0_calibration = _fit_t0_calibration(
        flow, flow_norm, m1, fiii, bank, query_model, query_fit,
        validation, val_b,
        raw_memory=(train_keys, train_actions, train_fit["episode_id"]),
        seed=seed + 839, device=device)
    fixed_t0 = t0_calibration["selected_fixed_t0"]
    low, high = t0_calibration["selected_entropy_low_high"]
    abs_low, abs_high = t0_calibration["selected_absolute_similarity_low_high"]
    rts_low, rts_high = t0_calibration["selected_rts_similarity_low_high"]

    test_qrows = _normal_query_rows(test)
    test_keys = _query_embedding(query_model, query_fit, test_qrows, device=device)
    test_b, _ = m1.encode(test["a"][:, :HORIZON], test["s"])
    initial_ids = np.arange(len(m6.prototypes),dtype=np.int64)
    # Stream test-suite values through M6, collecting each query before adding it.
    registered_before_stream = int(m6.registered_count)
    stream_meta, stream_order = _m6_new_suite_stream(
        m6, bank, test, test_b, test_keys,
        temperature=query_fit["temperature"], initial_unit_ids=initial_ids,
        train_keys=train_keys, train_actions=train_actions,
        train_episodes=train_fit["episode_id"],
        seed=seed + 853)
    # The stream helper's chunked unit snapshots make posterior membership
    # causal; evaluate all conditions using those snapshots, then insertions
    # above have already exercised M6. This pass replays saved per-row snapshots.
    # For the lightweight evaluator, the saved top-level unit posterior is the
    # stable source for every stochastic comparison on that row.
    run_rows = []
    sample_rng = np.random.default_rng(seed + 857)
    eval_mse = {k: [] for k in (
        "normal", "RAGDP", "RTS", "OptimusVLA_gaussian", "iii_top1",
        "component_fixed", "component_entropy", "component_absolute_similarity",
        "component_shuffled_t0", "blend_fixed", "blend_entropy",
        "shuffled_entropy", "gold_entropy")}
    eval_best = {k: [] for k in eval_mse}
    eval_grip = {k: {x:[] for x in GRIP_METRICS} for k in eval_mse}
    new_mse = {k: [] for k in eval_mse}
    new_best = {k: [] for k in eval_mse}
    new_grip = {k: {x:[] for x in GRIP_METRICS} for k in eval_mse}
    new_motion_flags, new_motion_registered = [], []
    entropy_by_row = {int(snap["row_index"]): float(entropy_t0(
        snap["entropy"], len(snap["unit_ids"]), low, high)) for snap in stream_meta}
    shuffled_t0_values = np.random.default_rng(seed + 859).permutation(
        np.asarray(list(entropy_by_row.values()), np.float32))
    shuffled_t0_by_row = dict(zip(entropy_by_row, shuffled_t0_values))
    novel_motion = _novel_motion_mask(test,split,np.unique(train_fit['skill_class']))
    # Restore dynamic bank state by replaying snapshots and correspondingly
    # preparing each query prediction. Saved unit centers/weights are causal.
    for snapshot in stream_meta:
        i = snapshot["row_index"]
        qrow = _subset(test, np.arange(len(test["a"])) == i)
        qkey = test_keys[i:i+1]
        qbase = test_b[i:i+1]
        target = test["a"][i:i+1, :HORIZON]
        ids, centers = snapshot["unit_ids"], snapshot["centers"]
        weights, sigmas = snapshot["weights"], snapshot["sigmas"]
        entropy = snapshot["entropy"]
        top_pos = int(weights.argmax())
        top_id = int(ids[top_pos])
        top_base = centers[top_pos]
        top_sigma = float(sigmas[top_pos])
        hnorm = entropy / max(np.log(len(ids)), 1e-8)
        adaptive_t0 = float(entropy_t0(entropy, len(ids), low, high))
        absolute_t0 = float(similarity_t0(
            snapshot["max_similarity"], *t0_calibration["absolute_similarity_reference"],
            abs_low, abs_high))
        rts_t0 = float(similarity_t0(
            snapshot["raw_similarity"], *t0_calibration["rts_similarity_reference"],
            rts_low, rts_high))
        shuffled_t0 = float(shuffled_t0_by_row[i])
        unit_rng = np.random.default_rng(seed * 1_000_003 + i + 907)
        def anchor_rng():
            return np.random.default_rng(seed * 1_000_003 + i + 911)
        sampled_pos = unit_rng.choice(len(ids), size=EVAL_SAMPLES, p=weights)
        sampled_b = centers[sampled_pos][:, None, :]
        sampled_sigma = sigmas[sampled_pos][:, None]
        sampled_sources = _draw_decoded_samples(
            m1, sampled_b, sampled_sigma, qrow, flow_norm, anchor_rng(),
            draws=EVAL_SAMPLES)
        eps8 = np.random.default_rng(seed * 1_000_003+i+919).standard_normal(
            (EVAL_SAMPLES,1,HORIZON*target.shape[-1])).astype(np.float32)
        eps8 = eps8.reshape(EVAL_SAMPLES, 1, HORIZON, target.shape[-1])
        t0_fixed_vec = np.asarray([fixed_t0], np.float32)
        t0_entropy_vec = np.asarray([adaptive_t0], np.float32)
        outputs = {}
        outputs["normal"] = _rollout_flow(
            flow, flow_norm, qrow, None, eps8.reshape(EVAL_SAMPLES, 1, -1),
            np.ones(1, np.float32), steps=10, device=device)
        raw_action = snapshot["raw_action"]
        raw_action_norm = _standardize(raw_action.reshape(1, -1),
                                       flow_norm["a_mean"], flow_norm["a_scale"])
        raw_src = np.broadcast_to(raw_action_norm[None],
                                  (EVAL_SAMPLES, 1, raw_action_norm.shape[-1]))
        outputs["RAGDP"] = _rollout_flow(
            flow, flow_norm, qrow, raw_src, eps8.reshape(EVAL_SAMPLES, 1, -1),
            np.asarray([t0_calibration["selected_ragdp_fixed_t0"]],np.float32), steps=10, device=device)
        outputs["RTS"] = _rollout_flow(
            flow, flow_norm, qrow, raw_src, eps8.reshape(EVAL_SAMPLES, 1, -1),
            np.asarray([rts_t0], np.float32), steps=10, device=device)
        outputs["component_fixed"] = _rollout_flow(
            flow, flow_norm, qrow, sampled_sources,
            eps8.reshape(EVAL_SAMPLES, 1, -1), t0_fixed_vec, steps=10, device=device)
        outputs["component_entropy"] = _rollout_flow(
            flow, flow_norm, qrow, sampled_sources,
            eps8.reshape(EVAL_SAMPLES, 1, -1), t0_entropy_vec, steps=10, device=device)
        outputs["component_absolute_similarity"] = _rollout_flow(
            flow, flow_norm, qrow, sampled_sources,
            eps8.reshape(EVAL_SAMPLES, 1, -1), np.asarray([absolute_t0], np.float32),
            steps=10, device=device)
        outputs["component_shuffled_t0"] = _rollout_flow(
            flow, flow_norm, qrow, sampled_sources,
            eps8.reshape(EVAL_SAMPLES, 1, -1), np.asarray([shuffled_t0], np.float32),
            steps=10, device=device)
        gaussian_mean, gaussian_variance = moment_match_bases(centers, sigmas, weights)
        gaussian_b = np.broadcast_to(gaussian_mean[None, None, :],
                                     (EVAL_SAMPLES, 1, len(gaussian_mean)))
        gaussian_std = np.broadcast_to(np.sqrt(gaussian_variance)[None, None, :],
                                       gaussian_b.shape)
        gaussian_sources = _draw_decoded_samples(
            m1, gaussian_b, gaussian_std, qrow, flow_norm, anchor_rng(),
            draws=EVAL_SAMPLES)
        outputs["OptimusVLA_gaussian"] = _rollout_flow(
            flow, flow_norm, qrow, gaussian_sources,
            eps8.reshape(EVAL_SAMPLES, 1, -1), np.asarray([absolute_t0], np.float32),
            steps=10, device=device)
        blend_b, blend_sigma, _ = _neighbor_blend(ids, centers, weights, sigmas, m6)
        blend_sources = _draw_decoded_samples(
            m1, np.broadcast_to(blend_b[None, None], (EVAL_SAMPLES, 1, len(blend_b))),
            np.full((EVAL_SAMPLES, 1), blend_sigma, np.float32),
            qrow, flow_norm, anchor_rng(), draws=EVAL_SAMPLES)
        outputs["blend_fixed"] = _rollout_flow(
            flow, flow_norm, qrow, blend_sources, eps8.reshape(EVAL_SAMPLES, 1, -1),
            t0_fixed_vec, steps=10, device=device)
        outputs["blend_entropy"] = _rollout_flow(
            flow, flow_norm, qrow, blend_sources, eps8.reshape(EVAL_SAMPLES, 1, -1),
            t0_entropy_vec, steps=10, device=device)
        # Permute key/value associations; uniform component draws would change
        # the posterior's weight profile rather than only its associations.
        shuffled_pos = unit_rng.permutation(len(ids))[sampled_pos]
        shuffled_b = centers[shuffled_pos][:, None]
        shuffled_sigma = sigmas[shuffled_pos][:, None]
        shuffled_sources = _draw_decoded_samples(m1, shuffled_b, shuffled_sigma,
                                                 qrow, flow_norm, anchor_rng(),
                                                 draws=EVAL_SAMPLES)
        outputs["shuffled_entropy"] = _rollout_flow(
            flow, flow_norm, qrow, shuffled_sources, eps8.reshape(EVAL_SAMPLES, 1, -1),
            t0_entropy_vec, steps=10, device=device)
        gold_sources = _draw_decoded_samples(
            m1, np.broadcast_to(qbase[None], (EVAL_SAMPLES, 1, qbase.shape[-1])),
            np.zeros((EVAL_SAMPLES,1),np.float32), qrow,
            flow_norm, anchor_rng(), draws=EVAL_SAMPLES)
        outputs["gold_entropy"] = _rollout_flow(
            flow, flow_norm, qrow, gold_sources, eps8.reshape(EVAL_SAMPLES, 1, -1),
            t0_entropy_vec, steps=10, device=device)
        top1_sources = _draw_decoded_samples(m1,top_base[None],np.zeros(1,np.float32),
                        qrow,flow_norm,anchor_rng(),draws=EVAL_SAMPLES)
        outputs["iii_top1"] = _rollout_flow(flow,flow_norm,qrow,top1_sources,
                        eps8.reshape(EVAL_SAMPLES,1,-1),t0_fixed_vec,steps=10,device=device)
        is_new_motion = bool(novel_motion[i])
        for name, prediction in outputs.items():
            metrics = _metrics(prediction, target)
            eval_mse[name].append(metrics["mse"])
            eval_best[name].append(metrics["best_of_8_mse"])
            for kind in GRIP_METRICS:
                eval_grip[name][kind].append(metrics["gripper_timing_tv"][kind])
            if is_new_motion:
                new_mse[name].append(metrics["mse"])
                new_best[name].append(metrics["best_of_8_mse"])
                for kind in GRIP_METRICS:
                    new_grip[name][kind].append(metrics["gripper_timing_tv"][kind])
        new_motion_flags.append(is_new_motion)
        new_motion_registered.append(bool(snapshot["selected_post_add"]) if is_new_motion else False)

    # Add new suite stream (the helper already performed the frozen M6 inserts).
    if (state_hash(m1) != m1_hash or state_hash(query_model) != e_hash
            or state_hash(flow) != flow_hash):
        raise AssertionError("M1, query tower, or flow head changed during new-suite memory insertion")
    n_new = int(np.sum(new_motion_flags))
    post_add_rate = _post_add_rate(new_motion_flags,new_motion_registered)
    result_rows = []
    for name in eval_mse:
        result_rows.append({"split": split, "seed": seed, "horizon": HORIZON,
                            "population": "all new-suite intervals",
                            "condition": name, "n": len(test["a"]),
                            "mse": float(np.mean(eval_mse[name])),
                            "best_of_8_mse": float(np.mean(eval_best[name])),
                            "gripper_timing_tv": {k: float(np.mean(v)) for k, v in eval_grip[name].items()},
                            "new_motion_post_add_top1": None})
        if new_mse[name]:
            result_rows.append({"split": split, "seed": seed, "horizon": HORIZON,
                                "population": "novel-motion intervals in four tasks",
                                "condition": name, "n": len(new_mse[name]),
                                "mse": float(np.mean(new_mse[name])),
                                "best_of_8_mse": float(np.mean(new_best[name])),
                                "gripper_timing_tv": {k: float(np.mean(v)) for k, v in new_grip[name].items()},
                                "new_motion_post_add_top1": (post_add_rate if name in ("component_entropy", "blend_entropy") else None)})
    mse_by_name = {r["condition"]: r["mse"] for r in result_rows
                   if r["population"] == "all new-suite intervals"}
    normal_mse = mse_by_name["normal"]
    ragdp_mse = mse_by_name["RAGDP"]
    star_mse = mse_by_name["component_entropy"]
    gates={}
    for population in {r['population'] for r in result_rows}:
        by_name={r['condition']:r for r in result_rows if r['population']==population}
        ordinary,star,raw,shuffled=(by_name[k] for k in ('normal','component_entropy','RAGDP','shuffled_entropy'))
        gates[population]={
            'relative_improvement_vs_normal':(ordinary['mse']-star['mse'])/max(ordinary['mse'],1e-8),
            'star_minus_RAGDP':star['mse']-raw['mse'],
            'shuffled_minus_normal':shuffled['mse']-ordinary['mse'],
            'best_of_8_minus_normal':star['best_of_8_mse']-ordinary['best_of_8_mse'],
            'gripper_minus_normal':{k:star['gripper_timing_tv'][k]-ordinary['gripper_timing_tv'][k] for k in GRIP_METRICS}}
    return {"split": split, "seed": seed, "horizon": HORIZON,
            "train_rows": len(train_fit["a"]), "validation_rows": len(validation["a"]),
            "prediction_covered_train_rows": int(prediction_covered_train_rows),
            "prediction_missing_train_rows": int(prediction_missing_train_rows),
            "smoke_selected_train_rows": int(len(train["episode_id"])),
            "query_rows": len(test["a"]), "excluded_unknown_class_rows": excluded,
            "M6_training_unit_count": int(len(initial_ids)),
            "M6_after_stream_unit_count": int(len(m6.prototypes)),
            "M6_epsilon": float(m6.epsilon),
            "M6_registered_on_stream": int(m6.registered_count - registered_before_stream),
            "M6_pending_after_stream": int(len(m6.pending)),
            "M1_epochs": int(m1.epochs_run),
            "F_epochs": int(flow_fit["best_epoch"]),
            "F_validation_loss": float(flow_fit["best_validation_loss"]),
            "iii_top1_uses_same_F":True,
            "query_encoder_epochs": int(query_fit["epochs"]),
            "query_encoder_temperature": float(query_fit["temperature"]),
            "t0_calibration": t0_calibration,
            "source_mix": list(source_mix), "rows": result_rows,
            "gate_point_estimates":gates,
            "s1a_relative_improvement": float((normal_mse - star_mse) / max(normal_mse, 1e-8)),
            "s1a_vs_ragdp": float((ragdp_mse - star_mse) / max(ragdp_mse, 1e-8)),
            "s1b_post_add_top1": post_add_rate,
            "s1b_new_motion_rows": n_new,
            "s1b_novel_motion_definition":"four Split-A tasks, excluding motion classes seen in training",
            "stream_order_seed": seed + 853,
            "duration_seconds": time.perf_counter() - tic}


def run(*, output_root=DEFAULT_OUTPUT_ROOT, cache_dir=DEFAULT_CACHE,
        dataset_dir=DEFAULT_DATASET, seeds=(0, 1, 2), splits=("A", "B"),
        device_name="cuda", source_mix=(0.65, 0.05, 0.30), smoke=False,
        smoke_train_episodes=3, smoke_eval_episodes=3,
        m1_epochs=60, f_epochs=80, fiii_epochs=80, batch_size=512,
        subtask_source="gold",
        predicted_root=Path("/work/.skillnet-state/outputs/memory-pilot/subtask-pi05-full-cache-20261028T072222Z")):
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device(device_name)
    start = time.perf_counter()
    rows = load_rows(cache_dir, dataset_dir, max_horizon=HORIZON)
    runs = []
    all_results = []
    for split in splits:
        for seed in seeds:
            result = _run_one(rows, split=split, seed=seed, device=device,
                              source_mix=source_mix, smoke=smoke,
                              smoke_train_episodes=smoke_train_episodes,
                              smoke_eval_episodes=smoke_eval_episodes,
                              m1_epochs=m1_epochs, f_epochs=f_epochs,
                              fiii_epochs=fiii_epochs, batch_size=batch_size,
                              subtask_source=subtask_source, cache_dir=cache_dir,
                              predicted_root=predicted_root)
            runs.append(result)
            all_results.extend(result["rows"])
            print(f"S1 split={split} seed={seed}: units={result['M6_training_unit_count']}→"
                  f"{result['M6_after_stream_unit_count']}; queries={result['query_rows']}; "
                  f"star MSE={next(r['mse'] for r in result['rows'] if r['condition']=='component_entropy'):.4f}",
                  flush=True)
    elapsed = time.perf_counter() - start
    summary = {
        "spec": "/work/docs/research/anchor_flow_skill_memory.md §2, §4 S1",
        "seed_list": list(seeds), "splits": list(splits), "horizon": HORIZON,
        "subtask_source": subtask_source,
        "predicted_subtask_files": {
            split: str(predicted_subtask_file(predicted_root, split)) for split in splits},
        "training_coverage_policy": "all valid H=25 rows in each training suite; predicted source is joined exactly by episode_id and step",
        "device": str(device), "smoke": bool(smoke),
        "source_mix_retrieved_anchor_gold_anchor_ordinary_path": list(source_mix),
        "anchor_start_formula": "b'~N(c_k,sigma_k^2 I); a_k=Dec(b',r,s); x_t0=(1-t0)*a_k+t0*epsilon",
        "flow_training_formula": "x_t=(1-t)*a+t*epsilon; ordinary loss ||F-(epsilon-A)||^2; augmented loss ||x_t-t*F-A||^2; same F for all comparisons",
        "comparisons": ["normal", "RAGDP", "RTS", "OptimusVLA_gaussian", "iii_top1",
                        "component_fixed", "component_entropy", "component_absolute_similarity",
                        "component_shuffled_t0", "blend_fixed", "blend_entropy",
                        "shuffled_entropy", "gold_entropy"],
        "language_free_query": {
            "gold": "E_img(k) + E_sub(gold plan segment, frozen pi05 token means) + E_state(proprio); f zeroed",
            "none": "E_img(k) + E_state(proprio); no instruction/subtask text; f zeroed",
            "instruction": "E_img(k) + E_sub(whole task instruction, frozen pi05 token means) + E_state(proprio); f zeroed",
            "predicted": "E_img(k) + E_sub(split-specific pi05 predicted segment, frozen pi05 token means) + E_state(proprio); f zeroed",
        }[subtask_source],
        "M6": {"kappa": 0.1, "min_support": 10, "min_episodes": 3,
               "functional_probe_count": 64, "update_centers_on_add": False},
        "flow_steps": 10, "evaluation_samples": EVAL_SAMPLES,
        "new_motion_task_indices_split_A": list(NEW_MOTION_TASKS_A),
        "runs": runs, "rows": all_results,
        "smoke_duration_seconds": elapsed if smoke else 0,
        "full_run_estimate": "pending smoke duration",
    }
    if smoke:
        estimates=[]
        for fit in runs:
            tasks=(*SPLITS[fit['split']][0],*SPLITS[fit['split']][1])
            full_n=int((rows['valid_h25'] & np.isin(rows['task_index'],tasks)).sum())
            sampled_n=fit['train_rows']+fit['validation_rows']+fit['query_rows']
            estimates.append(fit['duration_seconds']*full_n/max(sampled_n,1)/60)
        summary['full_run_estimate'] = f"{min(estimates):.1f}–{max(estimates):.1f} min per seed×split by actual row scaling; not a measured six-process runtime"
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / (datetime.now(timezone.utc).strftime("anchor-flow-%Y%m%dT%H%M%SZ")
                            + "-" + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    (output / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (output / "summary.md").write_text(_report(summary))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--splits", default="A,B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-mix", default="0.65,0.05,0.30",
                        help="fractions for retrieved-anchor,gold-b-anchor,ordinary path")
    parser.add_argument("--subtask-source", choices=("gold", "none", "instruction", "predicted"),
                        default="gold")
    parser.add_argument("--predicted-subtask-root", type=Path,
                        default=Path("/work/.skillnet-state/outputs/memory-pilot/subtask-pi05-full-cache-20261028T072222Z"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-episodes", type=int, default=3)
    parser.add_argument("--smoke-eval-episodes", type=int, default=3)
    parser.add_argument("--m1-epochs", type=int, default=60)
    parser.add_argument("--flow-epochs", type=int, default=80)
    parser.add_argument("--fiii-epochs", type=int, default=80,help="legacy option; iii_top1 now uses the shared F")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    splits = tuple(x.strip() for x in args.splits.split(",") if x.strip())
    if any(x not in SPLITS for x in splits):
        parser.error("splits must be A and/or B")
    output = run(output_root=args.output_root, cache_dir=args.cache_dir,
                 dataset_dir=args.dataset, seeds=seeds, splits=splits,
                 device_name=args.device, source_mix=parse_source_mix(args.source_mix),
                 smoke=args.smoke, smoke_train_episodes=args.smoke_train_episodes,
                 smoke_eval_episodes=args.smoke_eval_episodes,
                 m1_epochs=args.m1_epochs, f_epochs=args.flow_epochs,
                 fiii_epochs=args.fiii_epochs, batch_size=args.batch_size,
                 subtask_source=args.subtask_source,
                 predicted_root=args.predicted_subtask_root)
    print(f"output: {output}")


if __name__ == "__main__":
    main()
