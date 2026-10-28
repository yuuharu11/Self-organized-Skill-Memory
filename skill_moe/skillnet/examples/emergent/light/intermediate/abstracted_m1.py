"""Intervention (iii): M1 bases with residual resampling and an action denoiser."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..methods.m1 import M1
from .experiment import (
    DEFAULT_CACHE,
    DEFAULT_DATASET,
    DEFAULT_OUTPUT_ROOT,
    GOAL_TASK_INDICES,
    HORIZONS,
    T0_VALUES,
    TRAIN_TASK_INDICES,
    _episode_train_validation_indices,
    _onehot,
    _query_features,
    _query_key,
    _standardize,
    _standardize_fit,
    align_segment_labels,
    integrate_flow_sources,
    load_rows,
    retrieve_top1,
    retrieve_top1_same_class,
    split_split_a,
)


PREVIOUS_RESULTS = Path(
    "/work/.skillnet-state/outputs/memory-pilot/"
    "intermediate-action-20261027T122854Z-e207f4/results.json"
)
WHITE_NOISE_LEVELS = (0.0, 0.2, 0.4)
RESIDUAL_DRAWS = 4


def white_noise_mix(x_mid: np.ndarray, noise: np.ndarray, t0: float) -> np.ndarray:
    if not 0.0 <= t0 < 1.0:
        raise ValueError("white-noise t0 must be in [0,1)")
    return ((1.0 - t0) * np.asarray(x_mid, np.float32)
            + t0 * np.asarray(noise, np.float32)).astype(np.float32)


class AbstractedActionRegressor(nn.Module):
    """Same 2-hidden-layer width as F, conditioned on x_mid in addition to f,s."""

    def __init__(self, input_dim: int, action_dim: int, width: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, action_dim),
        )

    def forward(self, x_mid, feature, state):
        return self.net(torch.cat([x_mid, feature, state], dim=-1))


def action_motion_labels(actions: np.ndarray) -> np.ndarray:
    """Coarse direction, rotation-axis/sign, and gripper labels for action chunks."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[-1] < 7:
        raise ValueError("actions must have shape (batch, horizon, >=7)")

    def signed_axis(vector):
        magnitude = np.abs(vector)
        axis = magnitude.argmax(axis=-1)
        peak = magnitude.max(axis=-1)
        sign = np.sign(np.take_along_axis(vector, axis[..., None], axis=-1)[..., 0])
        # 0 is stationary; 1..6 are +/- x, +/- y, +/- z.
        return np.where(peak < 1e-4, 0, 1 + 2 * axis + (sign < 0)).astype(np.int8)

    translation = actions[..., :3].sum(axis=1)
    rotation = actions[..., 3:6].sum(axis=1)
    grip = actions[..., 6].mean(axis=1)
    grip_label = np.where(grip > 1e-4, 2, np.where(grip < -1e-4, 1, 0)).astype(np.int8)
    return np.stack([signed_axis(translation), signed_axis(rotation), grip_label], axis=-1)


def pairwise_label_agreement(draw_labels: np.ndarray) -> dict[str, float]:
    """Mean pairwise label agreement across residual draws, per label family."""
    labels = np.asarray(draw_labels)
    if labels.ndim != 3 or labels.shape[-1] != 3 or labels.shape[0] < 2:
        raise ValueError("draw_labels must be (draws>=2, rows, 3)")
    pairs = [(i, j) for i in range(labels.shape[0]) for j in range(i + 1, labels.shape[0])]
    names = ("direction", "rotation", "gripper")
    return {name: float(np.mean([labels[i, :, d] == labels[j, :, d] for i, j in pairs]))
            for d, name in enumerate(names)}


def agreement_to_reference(draw_labels: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    labels = np.asarray(draw_labels)
    reference = np.asarray(reference)
    names = ("direction", "rotation", "gripper")
    if labels.ndim != 3 or labels.shape[-1] != 3 or reference.shape != labels.shape[1:]:
        raise ValueError("reference labels do not match draw label shape")
    return {name: float(np.mean(labels[:, :, d] == reference[None, :, d]))
            for d, name in enumerate(names)}


def _decode_residual_draws(m1: M1, bases: np.ndarray, feature: np.ndarray, state: np.ndarray,
                           *, draws: int, seed: int, batch_size: int = 768) -> np.ndarray:
    bases = np.asarray(bases, np.float32)
    rng = np.random.default_rng(seed)
    n = len(bases)
    output = np.empty((draws, n, *m1.action_shape), dtype=np.float32)
    for draw in range(draws):
        residual = rng.standard_normal((n, m1.residual_dim)).astype(np.float32)
        for lo in range(0, n, batch_size):
            hi = min(n, lo + batch_size)
            output[draw, lo:hi] = m1.decode(
                bases[lo:hi], residual[lo:hi], feature[lo:hi], state[lo:hi])
    return output


def _fit_abstracted_regressor(m1: M1, train: dict[str, np.ndarray], actions: np.ndarray,
                              bases: np.ndarray, mids: np.ndarray, *, epochs: int,
                              batch_size: int, patience: int, seed: int,
                              device: torch.device) -> tuple[AbstractedActionRegressor, dict[str, Any]]:
    train_ix, valid_ix = _episode_train_validation_indices(train, seed + 17)
    a_mean, a_scale = m1.am, m1.ass
    target = _standardize(actions.reshape(len(actions), -1), a_mean, a_scale)
    mids_flat = mids.reshape(mids.shape[0], mids.shape[1], -1)
    mid_norm = _standardize(mids_flat.reshape(-1, mids_flat.shape[-1]), a_mean, a_scale).reshape(mids_flat.shape)
    f = _standardize(train["f"], m1.fm, m1.fs)
    s = _standardize(train["s"], m1.sm, m1.ss)
    ft = torch.as_tensor(f, dtype=torch.float32, device=device)
    st = torch.as_tensor(s, dtype=torch.float32, device=device)
    at = torch.as_tensor(target, dtype=torch.float32, device=device)
    midt = torch.as_tensor(mid_norm, dtype=torch.float32, device=device)
    train_ix_t = torch.as_tensor(train_ix, dtype=torch.long, device=device)
    valid_ix_t = torch.as_tensor(valid_ix, dtype=torch.long, device=device)
    rng = torch.Generator(device=device).manual_seed(seed + 101)
    val_rng = torch.Generator(device=device).manual_seed(seed + 202)
    action_dim = target.shape[-1]
    model = AbstractedActionRegressor(f.shape[-1] + s.shape[-1] + action_dim,
                                      action_dim, width=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    levels = torch.as_tensor(WHITE_NOISE_LEVELS, device=device, dtype=torch.float32)
    # Fixed validation noise makes each epoch comparable; residual draws are
    # independent of training draws and shared across the three noise levels.
    valid_noise = torch.randn((len(WHITE_NOISE_LEVELS), RESIDUAL_DRAWS,
                               len(valid_ix), action_dim), generator=val_rng, device=device)
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(epochs):
        order = train_ix_t[torch.randperm(len(train_ix_t), generator=rng, device=device)]
        model.train()
        batch_losses = []
        for start in range(0, len(order), batch_size):
            ix = order[start:start + batch_size]
            draw_ix = torch.randint(0, mids.shape[0], (len(ix),), generator=rng, device=device)
            mid = midt[draw_ix, ix]
            level_ix = torch.randint(0, len(WHITE_NOISE_LEVELS), (len(ix),), generator=rng, device=device)
            level = levels[level_ix, None]
            noise = torch.randn(mid.shape, generator=rng, device=device)
            noisy_mid = (1.0 - level) * mid + level * noise
            prediction = model(noisy_mid, ft[ix], st[ix])
            loss = F.mse_loss(prediction, at[ix])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for li, level_value in enumerate(WHITE_NOISE_LEVELS):
                for draw in range(RESIDUAL_DRAWS):
                    for start in range(0, len(valid_ix), 2048):
                        lo, hi = start, min(len(valid_ix), start + 2048)
                        ix = valid_ix_t[lo:hi]
                        mid = midt[draw, ix]
                        noise = valid_noise[li, draw, lo:hi]
                        noisy_mid = (1.0 - level_value) * mid + level_value * noise
                        prediction = model(noisy_mid, ft[ix], st[ix])
                        valid_losses.append(float(F.mse_loss(prediction, at[ix]).cpu()))
        train_loss = float(np.mean(batch_losses))
        valid_loss = float(np.mean(valid_losses))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": valid_loss})
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_epoch = epoch + 1
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"history": history, "best_epoch": best_epoch,
                   "best_validation_loss": best_loss,
                   "final_validation_loss": history[-1]["validation_loss"],
                   "early_stopped": len(history) < epochs,
                   "train_rows": int(len(train_ix)), "validation_rows": int(len(valid_ix)),
                   "residual_draws_for_validation": RESIDUAL_DRAWS,
                   "white_noise_levels": list(WHITE_NOISE_LEVELS)}


def _predict_abstracted(m1: M1, model: AbstractedActionRegressor,
                        bases: np.ndarray, feature: np.ndarray, state: np.ndarray,
                        *, t0: float, seed: int, draws: int = RESIDUAL_DRAWS,
                        batch_size: int = 768) -> np.ndarray:
    decoded = _decode_residual_draws(m1, bases, feature, state, draws=draws, seed=seed)
    normalized_mids = _standardize(decoded.reshape(-1, decoded.shape[-2] * decoded.shape[-1]),
                                   m1.am, m1.ass).reshape(draws, len(bases), -1)
    fn = torch.as_tensor(_standardize(feature, m1.fm, m1.fs), device=model.net[0].weight.device)
    sn = torch.as_tensor(_standardize(state, m1.sm, m1.ss), device=model.net[0].weight.device)
    device = fn.device
    action_dim = normalized_mids.shape[-1]
    rng = np.random.default_rng(seed + 9001)
    noises = rng.standard_normal((draws, len(bases), action_dim)).astype(np.float32)
    mixed = white_noise_mix(normalized_mids, noises, t0)
    preds = np.empty_like(mixed)
    model.eval()
    with torch.no_grad():
        for draw in range(draws):
            for lo in range(0, len(bases), batch_size):
                hi = min(len(bases), lo + batch_size)
                x = torch.as_tensor(mixed[draw, lo:hi], device=device)
                out = model(x, fn[lo:hi], sn[lo:hi]).cpu().numpy()
                preds[draw, lo:hi] = out
    return (preds * m1.ass + m1.am).reshape(draws, len(bases), *m1.action_shape)


def _mse(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, np.float32)
    target = np.asarray(target, np.float32)
    return float(np.mean(np.square(prediction - target[None])))


def _same_class_alternative_indices(train_classes, query_classes, selected_indices, seed):
    """Pick a different stored interval from the same annotated subtask."""
    rng = np.random.default_rng(seed)
    train_classes = np.asarray(train_classes)
    query_classes = np.asarray(query_classes)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    alternatives = np.full(len(query_classes), -1, dtype=np.int64)
    for row, (label, selected) in enumerate(zip(query_classes, selected_indices)):
        candidates = np.flatnonzero(train_classes == label)
        candidates = candidates[candidates != selected]
        if len(candidates):
            alternatives[row] = int(rng.choice(candidates))
    return alternatives


def _masked_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(target)
    return float(np.square(np.asarray(prediction)[..., mask] - np.asarray(target)[mask]).mean())


def _motion_agreement(m1: M1, bases: np.ndarray, feature: np.ndarray, state: np.ndarray,
                      reference_actions: np.ndarray, *, seed: int, known_mask: np.ndarray) -> dict[str, Any]:
    decoded = _decode_residual_draws(m1, bases, feature, state, draws=RESIDUAL_DRAWS,
                                     seed=seed, batch_size=768)
    ref_labels = action_motion_labels(reference_actions)
    labels = action_motion_labels(decoded.reshape(-1, *m1.action_shape)).reshape(
        RESIDUAL_DRAWS, len(bases), 3)
    groups = {"all": np.ones(len(bases), bool), "known_motion": known_mask,
              "new_motion": ~known_mask}
    result = {}
    for group, mask in groups.items():
        if not mask.any():
            result[group] = None
            continue
        result[group] = {"pairwise_agreement": pairwise_label_agreement(labels[:, mask]),
                         "agreement_with_demo": agreement_to_reference(labels[:, mask], ref_labels[mask]),
                         "rows": int(mask.sum())}
    result["draws_per_base"] = RESIDUAL_DRAWS
    result["label_definition"] = {
        "direction": "dominant signed axis of summed xyz action; 0=stationary, 1..6=+/-x,+/-y,+/-z",
        "rotation": "dominant signed axis of summed rotational action; same encoding",
        "gripper": "sign of mean gripper action; 0=neutral, 1=negative, 2=positive",
    }
    return result


def _prepare_m1_train(train: dict[str, np.ndarray], actions: np.ndarray, *, seed: int,
                      epochs: int, batch_size: int, device: torch.device) -> M1:
    m1 = M1(independence_weight=0.01, beta_r=1.0, width=128,
            epochs=epochs, batch_size=batch_size, device=str(device), seed_value=seed,
            decoder_obs=False)
    fit_data = {"actions": actions, "state": train["s"], "feature": train["f"],
                "key": train["k"], "episode_id": train["episode_id"],
                # The feature cache does not carry LIBERO initial-state IDs;
                # episode identity is the conservative memory exclusion group.
                "init_state_id": train["episode_id"],
                "seed": seed, "early_stopping_patience": 8}
    m1.fit(fit_data)
    return m1


def _run_horizon(train, goal, *, horizon: int, seed: int, m1_epochs: int,
                 action_epochs: int, batch_size: int, patience: int,
                 device: torch.device) -> dict[str, Any]:
    train_a = train["a"][:, :horizon].copy()
    goal_a = goal["a"][:, :horizon].copy()
    m1 = _prepare_m1_train(train, train_a, seed=seed + horizon,
                           epochs=m1_epochs, batch_size=512, device=device)
    base_train, _ = m1.encode(train_a, train["s"])
    base_goal, _ = m1.encode(goal_a, goal["s"])
    train_mids = _decode_residual_draws(m1, base_train, train["f"], train["s"],
                                        draws=RESIDUAL_DRAWS, seed=seed + horizon * 1000 + 1)
    valid_split, _ = _episode_train_validation_indices(train, seed + horizon * 11)
    del valid_split
    action_model, action_fit = _fit_abstracted_regressor(
        m1, train, train_a, base_train, train_mids, epochs=action_epochs,
        batch_size=batch_size, patience=patience, seed=seed + horizon * 100 + 3,
        device=device)

    class_ids = np.unique(np.concatenate([train["skill_class"], goal["skill_class"]]))
    task_train = _onehot(train["skill_class"], class_ids)
    task_goal = _onehot(goal["skill_class"], class_ids)
    key_mean, key_scale = _standardize_fit(_query_key(train))
    q_train = {variant: _query_features(train, task_train, variant, key_mean, key_scale)
               for variant in ("none", "concat", "same_class")}
    q_goal = {variant: _query_features(goal, task_goal, variant, key_mean, key_scale)
              for variant in ("none", "concat", "same_class")}
    dummy_units = np.zeros(len(train_a), dtype=np.int64)
    matches = {}
    for variant in ("none", "concat", "same_class"):
        if variant == "same_class":
            _, _, ix, unmatched = retrieve_top1_same_class(
                q_train[variant], base_train, dummy_units, q_goal[variant],
                train["skill_class"], goal["skill_class"])
        else:
            _, _, ix = retrieve_top1(q_train[variant], base_train, dummy_units, q_goal[variant])
            unmatched = np.zeros(len(goal_a), bool)
        matches[variant] = (ix, unmatched)

    known_mask = np.isin(goal["skill_class"], np.unique(train["skill_class"]))
    random_ix = np.random.default_rng(seed + horizon * 777).integers(0, len(base_train), len(base_goal))
    random_base = base_train[random_ix]
    noise_seed = seed + horizon * 9871
    results = []
    for variant in ("none", "concat", "same_class"):
        indices, unmatched = matches[variant]
        if variant == "same_class":
            eligible = ~unmatched
        else:
            eligible = np.ones(len(goal_a), bool)
        eval_indices = np.flatnonzero(eligible)
        if not len(eval_indices):
            raise ValueError(f"no eligible goal queries for retrieval mode {variant}")
        sources = {
            "memory_top1": base_train[indices[eval_indices]],
            "oracle_base": base_goal[eval_indices],
            "random_train_base": random_base[eval_indices],
        }
        target = goal_a[eval_indices]
        query_rows = {key: value[eval_indices] for key, value in goal.items()}
        for t0 in WHITE_NOISE_LEVELS:
            mse = {}
            mse_eligible = {}
            mse_new_class = {}
            for source, bases in sources.items():
                prediction = _predict_abstracted(
                    m1, action_model, bases, query_rows["f"], query_rows["s"],
                    t0=t0, seed=noise_seed + int(t0 * 1000) * 17,
                )
                mse[source] = _mse(prediction, target)
                common = known_mask[eval_indices]
                mse_eligible[source] = _mse(prediction[:, common], target[common])
                mse_new_class[source] = (_mse(prediction[:, ~common], target[~common])
                                         if (~common).any() else None)
            alternate_indices = _same_class_alternative_indices(
                train["skill_class"], goal["skill_class"][eval_indices],
                indices[eval_indices], seed + horizon * 4001 + int(variant == "concat") * 11
                + int(variant == "same_class") * 23)
            alt_rows = np.flatnonzero(alternate_indices >= 0)
            shuffle_stats = {}
            if len(alt_rows):
                shuffle_seed = noise_seed + int(t0 * 1000) * 17
                original_prediction = _predict_abstracted(
                    m1, action_model, base_train[indices[eval_indices][alt_rows]],
                    query_rows["f"][alt_rows], query_rows["s"][alt_rows],
                    t0=t0, seed=shuffle_seed)
                shuffled_prediction = _predict_abstracted(
                    m1, action_model, base_train[alternate_indices[alt_rows]],
                    query_rows["f"][alt_rows], query_rows["s"][alt_rows],
                    t0=t0, seed=shuffle_seed)
                alt_common = known_mask[eval_indices][alt_rows]
                for group, group_mask in {
                    "all": np.ones(len(alt_rows), dtype=bool),
                    "known_class": alt_common,
                    "unseen_class": ~alt_common,
                }.items():
                    if group_mask.any():
                        shuffle_stats[group] = {
                            "rows": int(group_mask.sum()),
                            "retrieved_mse": _mse(original_prediction[:, group_mask], target[alt_rows][group_mask]),
                            "shuffled_mse": _mse(shuffled_prediction[:, group_mask], target[alt_rows][group_mask]),
                        }
            results.append({
                "horizon": horizon, "retrieval": variant, "white_noise_t0": t0,
                "goal_rows": int(len(goal_a)), "mse_rows": int(len(eval_indices)),
                "eligible_query_rows": int(known_mask.sum()),
                "no_class_rows": int(unmatched.sum()),
                "base_memory_rows": int(len(base_train)),
                "mse": mse, "mse_eligible": mse_eligible,
                "mse_new_class": mse_new_class,
                "shuffle_same_class": shuffle_stats,
                "memory_minus_random": mse["memory_top1"] - mse["random_train_base"],
                "memory_minus_random_eligible": (mse_eligible["memory_top1"]
                                                 - mse_eligible["random_train_base"]),
                "memory_minus_oracle": mse["memory_top1"] - mse["oracle_base"],
                "memory_item_indices": indices.tolist(),
            })

    # For the residual sensitivity check, keep b fixed to the goal action's
    # encoded base and resample only r. This is evaluation-only.
    decoded_noise = _decode_residual_draws(m1, base_goal, goal["f"], goal["s"],
                                          draws=RESIDUAL_DRAWS, seed=noise_seed + 555)
    raw_label_draws = action_motion_labels(decoded_noise.reshape(-1, *m1.action_shape)).reshape(
        RESIDUAL_DRAWS, len(goal_a), 3)
    raw_reference = action_motion_labels(goal_a)
    recovered_draws = _predict_abstracted(m1, action_model, base_goal, goal["f"], goal["s"],
                                          t0=0.0, seed=noise_seed + 555)
    recovered_label_draws = action_motion_labels(recovered_draws.reshape(-1, *m1.action_shape)).reshape(
        RESIDUAL_DRAWS, len(goal_a), 3)
    diagnostics = {}
    for group, mask in {"all": np.ones(len(goal_a), bool), "known_motion": known_mask,
                        "new_motion": ~known_mask}.items():
        if not mask.any():
            diagnostics[group] = None
            continue
        diagnostics[group] = {
            "rows": int(mask.sum()),
            "decoded_pairwise_agreement": pairwise_label_agreement(raw_label_draws[:, mask]),
            "decoded_agreement_with_demo": agreement_to_reference(raw_label_draws[:, mask], raw_reference[mask]),
            "recovered_pairwise_agreement": pairwise_label_agreement(recovered_label_draws[:, mask]),
            "recovered_agreement_with_demo": agreement_to_reference(recovered_label_draws[:, mask], raw_reference[mask]),
        }
    return {
        "horizon": horizon,
        "train_rows": int(len(train_a)), "goal_rows": int(len(goal_a)),
        "goal_class_eligible_rows": int(known_mask.sum()),
        "m1": {"config": m1.describe(), "epochs_run": int(m1.epochs_run),
               "independence_weight": 0.01, "beta_r": 1.0,
               "training_rows": int(len(train_a)), "key_scope": "task_index 20-39 only"},
        "abstracted_action_head": action_fit,
        "results": results,
        "residual_motion_diagnostic": diagnostics,
    }


def _mean(rows, key):
    return float(np.mean([row[key] for row in rows])) if rows else float("nan")


def _combined_report(summary: dict[str, Any], previous: dict[str, Any]) -> str:
    lines = ["# Intermediate action experiment: (i), (ii), (iii)", "",
             "(i)/(ii) reuse the completed inter2 run listed in the manifest. (iii) trains M1 on object+spatial only and retrieves a continuous base; goal is query-only.", "",
             "## Held-out action MSE", "",
             "(ii) rows use top-1 `(k,s)` retrieval without subtask (a); DCT rows use target t0=0.8. (iii) reports top-1 b retrieval plus true-b and random-train-b bounds. Same-class (c) MSE is on matching goal classes only.", "",
             "| H | method | query | representation / t0 | MSE rows | action MSE | random-b/v | oracle / normal F |",
             "|---:|---|---|---|---:|---:|---:|---:|"]
    for horizon in HORIZONS:
        prior_rows = [r for r in previous["results"] if r["horizon"] == horizon]
        normal_f = prior_rows[0]["mse"]["normal"]
        full_n = prior_rows[0]["goal_rows"]
        lines.append(f"| {horizon} | (i) | — | normal F | {full_n} | {normal_f:.4f} | — | — |")
        for value_kind, target_t0 in (("raw", None), ("dct_lowpass", 0.8)):
            for eval_t0 in T0_VALUES:
                row = next((r for r in prior_rows if r["retrieval"] == "none"
                            and r["value_kind"] == value_kind and r["target_t0"] == target_t0
                            and r["t0"] == eval_t0), None)
                if row is None:
                    continue
                value_name = "raw A" if value_kind == "raw" else f"DCT target {target_t0:.1f}"
                lines.append(f"| {horizon} | (ii) | (a) none | {value_name}, eval {eval_t0:.1f} | "
                             f"{row['goal_rows']} | {row['mse']['memory_top1']:.4f} | "
                             f"{row['mse']['random_train']:.4f} | {row['mse']['oracle']:.4f} |")
        for row in summary["results"]:
            if row["horizon"] != horizon:
                continue
            mse = row["mse"]
            lines.append(f"| {horizon} | (iii) | ({row['retrieval']}) | white-noise t0={row['white_noise_t0']:.1f} | "
                         f"{row['mse_rows']}/{row['goal_rows']} | {mse['memory_top1']:.4f} | "
                         f"{mse['random_train_base']:.4f} | {mse['oracle_base']:.4f} |")
    lines += ["", "## Query-mode comparison on common goal rows", "",
              "All three modes are evaluated on goal rows whose eval_skill_class appears in the object+spatial memory, so the comparison does not reward (c) for dropping unmatched classes.", "",
              "| H | query | eligible rows | t0 | memory b | random b | oracle b | memory-random |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for horizon in HORIZONS:
        for variant in ("none", "concat", "same_class"):
            group = [r for r in summary["results"] if r["horizon"] == horizon
                     and r["retrieval"] == variant and r["white_noise_t0"] == 0.2]
            if group:
                row = group[0]
                m = row["mse_eligible"]
                lines.append(f"| {horizon} | {variant} | {row['eligible_query_rows']} | 0.2 | "
                             f"{m['memory_top1']:.4f} | {m['random_train_base']:.4f} | "
                             f"{m['oracle_base']:.4f} | {row['memory_minus_random_eligible']:+.4f} |")
    lines += ["", "## Fixed-base residual resampling diagnostic", "",
              "For each goal example, b is encoded once and held fixed while four independent r~N(0,I) draws are decoded. Pairwise agreement is among the resulting coarse labels; demo agreement compares each draw with the original action chunk.", "",
              "| H | subset | rows | decoded direction | decoded rotation | decoded gripper | recovered direction | recovered rotation | recovered gripper | decoded vs demo D/R/G |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for run in summary["horizons"]:
        for subset in ("all", "known_motion", "new_motion"):
            diag = run["residual_motion_diagnostic"].get(subset)
            if diag is None:
                continue
            d = diag["decoded_pairwise_agreement"]
            rec = diag["recovered_pairwise_agreement"]
            truth = diag["decoded_agreement_with_demo"]
            lines.append(f"| {run['horizon']} | {subset} | {diag['rows']} | {d['direction']:.3f} | "
                         f"{d['rotation']:.3f} | {d['gripper']:.3f} | {rec['direction']:.3f} | "
                         f"{rec['rotation']:.3f} | {rec['gripper']:.3f} | "
                         f"{truth['direction']:.3f}/{truth['rotation']:.3f}/{truth['gripper']:.3f} |")
    lines += ["", "## Fiii validation fit", "",
              "| H | M1 epochs | Fiii epochs | Fiii best epoch | best validation MSE | final validation MSE | stopped early |",
              "|---:|---:|---:|---:|---:|---:|:---:|"]
    for run in summary["horizons"]:
        fit = run["abstracted_action_head"]
        lines.append(f"| {run['horizon']} | {run['m1']['epochs_run']} | {len(fit['history'])} | "
                     f"{fit['best_epoch']} | {fit['best_validation_loss']:.5f} | "
                     f"{fit['final_validation_loss']:.5f} | {'yes' if fit['early_stopped'] else 'no'} |")
    lines += ["", "## Fit and scope", "",
              f"Seed {summary['seed']}; Fiii is a 2-hidden-layer width-256 regressor, trained with t0 sampled from {WHITE_NOISE_LEVELS}, episode-heldout validation, and early stopping. M1 uses HSIC 0.01, beta_r=1.0 and trains on task_index 20-39. Goal rows are never added to memory.", "",
              "Direction and rotation labels are the dominant signed axis of the summed action chunks; gripper is the sign of the mean gripper command. These are coarse diagnostics, not environment success metrics.", "",
              "H=10 DCT target t0=0.8 is absent from (ii) because the signal-dominant coefficient mask was empty.", ""]
    return "\n".join(lines)


def run(*, output_root: Path = DEFAULT_OUTPUT_ROOT, cache_dir: Path = DEFAULT_CACHE,
        dataset_dir: Path = DEFAULT_DATASET, previous_results: Path = PREVIOUS_RESULTS,
        seed: int = 0, m1_epochs: int = 60, action_epochs: int = 150,
        batch_size: int = 512, patience: int = 8, device_name: str = "cuda") -> Path:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device(device_name)
    previous = json.loads(previous_results.read_text())
    all_rows = load_rows(cache_dir, dataset_dir, max_horizon=25)
    train_all, goal_all = split_split_a(all_rows)
    horizons = []
    iii_results = []
    diagnostics = []
    for horizon in HORIZONS:
        train_mask = train_all[f"valid_h{horizon}"]
        goal_mask = goal_all[f"valid_h{horizon}"]
        train = {key: value[train_mask] for key, value in train_all.items()}
        goal = {key: value[goal_mask] for key, value in goal_all.items()}
        result = _run_horizon(train, goal, horizon=horizon, seed=seed,
                              m1_epochs=m1_epochs, action_epochs=action_epochs,
                              batch_size=batch_size, patience=patience, device=device)
        iii_results.extend(result["results"])
        horizons.append(result)
        diagnostics.append(result["residual_motion_diagnostic"])
        print(f"H={horizon}: M1 epochs={result['m1']['epochs_run']}; "
              f"Fiii best epoch={result['abstracted_action_head']['best_epoch']} "
              f"val={result['abstracted_action_head']['best_validation_loss']:.5f}", flush=True)

    run_id = datetime.now(timezone.utc).strftime("intermediate-action-iii-%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    out = output_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    summary = {
        "seed": seed,
        "split": "A: task_index 20-39 train; 10-19 goal held-out/query-only",
        "cache": str(cache_dir), "dataset": str(dataset_dir),
        "previous_inter2_results": str(previous_results),
        "m1_fit_config": {"name": "M1", "hsic": 0.01, "beta_r": 1.0,
                          "epochs_max": m1_epochs, "batch_size": 512,
                          "episode_id_used_as_init_state_exclusion_group": True},
        "iii_action_regressor_config": {"hidden_width": 256, "hidden_layers": 2,
                                        "epochs_max": action_epochs, "batch_size": batch_size,
                                        "early_stopping_patience": patience,
                                        "residual_draws": RESIDUAL_DRAWS,
                                        "white_noise_levels": list(WHITE_NOISE_LEVELS)},
        "goal_added_to_memory": False,
        "horizons": horizons,
        "results": iii_results,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out / "report.md").write_text(_combined_report(summary, previous))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--previous-results", type=Path, default=PREVIOUS_RESULTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--m1-epochs", type=int, default=60)
    parser.add_argument("--action-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    out = run(output_root=args.output_root, cache_dir=args.cache_dir,
              dataset_dir=args.dataset, previous_results=args.previous_results,
              seed=args.seed, m1_epochs=args.m1_epochs, action_epochs=args.action_epochs,
              batch_size=args.batch_size, patience=args.patience, device_name=args.device)
    print(f"output: {out}")


if __name__ == "__main__":
    main()
