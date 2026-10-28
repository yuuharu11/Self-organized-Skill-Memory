"""Geometric memory-query ablation for LIBERO skill retrieval.

Smoke: ``cd /work && source env/activate.sh && XLA_PYTHON_CLIENT_PREALLOCATE=false python -m skill_moe.skillnet.examples.emergent.light.intermediate.geo_query --smoke``

Full run (both suite splits):
``cd /work && source env/activate.sh && XLA_PYTHON_CLIENT_PREALLOCATE=false python -m skill_moe.skillnet.examples.emergent.light.intermediate.geo_query --device cuda``

The target point is an oracle: use the hand position at the first gripper
command sign change after the query start, searching past the action horizon
to the end of the current annotated skill segment (or episode if that is the
last segment). Rows with no later switch have no target and are excluded from
geometric-query training/evaluation. This hindsight target is an upper-bound
diagnostic and is not available at inference time.
"""
from __future__ import annotations

import argparse
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

from .abstracted_m1 import (
    _decode_residual_draws, _fit_abstracted_regressor,
    _predict_abstracted,
)
from .clip_retrieval import CLIP_EPOCHS, CLIP_PATIENCE, fit_dual_encoder
from .experiment import (
    DEFAULT_DATASET, DEFAULT_OUTPUT_ROOT, _standardize, _standardize_fit,
    integrate_flow_sources, load_rows, train_flow,
)
from ..methods.m6 import M6

HORIZON = 25
DEFAULT_CACHE = Path("/work/.skillnet-state/outputs/memory-pilot/emergent-light-heldout-20261026T061907Z-f96a16")
SPLITS = {
    "A": (tuple(range(20, 40)), tuple(range(10, 20))),
    "B": (tuple(range(20, 30)) + tuple(range(10, 20)), tuple(range(30, 40))),
}
T0S = (0.0, 0.2, 0.4)  # F_iii's training noise levels; no extrapolation.
TOPK = 5
SOFTMAX_TEMPERATURE = 0.25


def oracle_geometric_targets(rows: dict[str, np.ndarray], *, cache_dir: Path,
                             horizon: int = HORIZON, dataset_dir: Path | None = None):
    """Return next-command-switch targets, opening width, censor and offsets.

    The seven-dimensional cached actions have a discrete gripper command in
    column 6 (observed values are -1/+1). Proprio columns 6:8 are the two
    mirrored finger joint positions, not an aperture scalar; opening width is
    their difference, qpos[6] - qpos[7]. The action switch defines the event,
    while full-rate env_state supplies its corresponding hand position.
    """
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        manifest_path = Path(cache_dir) / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    paths = {str(ep["episode_id"]): Path(ep["cache"]) for ep in manifest["episodes"]}
    rel = np.full((len(rows["episode_id"]), 3), np.nan, np.float32)
    grip_state = np.zeros((len(rows["episode_id"]), 1), np.float32)
    censored = np.zeros(len(rows["episode_id"]), bool)
    offsets = np.full(len(rows["episode_id"]), np.nan, np.float32)
    loaded = {}
    if dataset_dir is not None:
        import pyarrow.parquet as pq
        parquet_paths = {int(p.stem.split("_")[-1]): p
                         for p in (Path(dataset_dir) / "data").rglob("episode_*.parquet")}
    for i, (episode, frame) in enumerate(zip(rows["episode_id"], rows["frame"])):
        episode = str(episode)
        if episode not in loaded:
            with np.load(paths[episode], allow_pickle=False) as data:
                env_step = np.asarray(data.get("env_step", np.arange(len(data["action"]))), np.int64)
                state = np.asarray(data.get("env_state", data["proprio"]), np.float32)
                action = np.asarray(data["action"], np.float32)
                sampled_step = np.asarray(data["step"], np.int64)
            segments = None
            if dataset_dir is not None:
                table = pq.read_table(parquet_paths[int(episode.rsplit("-", 1)[-1])],
                                      columns=["frame_index", "eval_skill_segment"])
                labels = dict(zip(table["frame_index"].to_pylist(), table["eval_skill_segment"].to_pylist()))
                segments = np.asarray([labels[int(t)] for t in env_step])
            loaded[episode] = (sampled_step, env_step, state, action, segments)
        sampled_step, env_step, state, action, segments = loaded[episode]
        if int(frame) not in set(sampled_step.tolist()):
            raise KeyError(f"cached frame {frame} missing for {episode}")
        start = int(np.searchsorted(env_step, int(frame)))
        if start >= len(env_step) or env_step[start] != int(frame):
            raise KeyError(f"frame {frame} absent from full-rate cache for {episode}")
        segment_end = int(env_step[-1])
        if segments is not None:
            changed = np.flatnonzero(segments[start:] != segments[start])
            if len(changed):
                segment_end = int(env_step[start + changed[0]]) - 1
        elif "skill_segment" in rows:
            same = ((rows["episode_id"] == episode) & (rows["frame"] > frame)
                    & (rows["skill_segment"] != rows["skill_segment"][i]))
            next_frames = rows["frame"][same]
            if len(next_frames):
                segment_end = min(segment_end, int(next_frames.min()) - 1)
        # A switch is a change between consecutive nonzero command signs. The
        # cache uses discrete +/-1 commands, so no tunable qpos threshold is
        # needed; zero values, if present, are ignored.
        command = np.sign(action[:, 6]).astype(np.int8)
        nonzero = np.flatnonzero(command != 0)
        switches = nonzero[1:][command[nonzero[1:]] != command[nonzero[:-1]]]
        start_frame = int(frame)
        end_frame = min(segment_end, int(env_step[-1]))
        valid_switches = switches[(env_step[switches] > start_frame) & (env_step[switches] <= end_frame)]
        grip_state[i, 0] = float(state[start, 6] - state[start, 7])
        if len(valid_switches):
            target_ix = int(valid_switches[0])
            rel[i] = state[target_ix, :3] - state[start, :3]
            offsets[i] = float(env_step[target_ix] - start_frame)
        else:
            censored[i] = True
    return rel, grip_state, censored, offsets


def _scaled_query(parts, means, scales):
    return np.concatenate([_standardize(x, m, s) for x, m, s in zip(parts, means, scales)], axis=-1)


def _kind_onehot(values, class_ids):
    lookup = {int(label): i for i, label in enumerate(class_ids)}
    result = np.zeros((len(values), len(class_ids)), np.float32)
    for row, label in enumerate(np.asarray(values, np.int64)):
        if int(label) in lookup:
            result[row, lookup[int(label)]] = 1.0
    return result


def geometric_query_variants(action_kind, rel_target, grip_state, means=None, scales=None):
    """Build Q1 (kind, relative xyz, grip), Q2 (xyz, grip), and Q3 (xyz)."""
    kind = np.asarray(action_kind, np.float32)
    rel_target = np.asarray(rel_target, np.float32)
    grip_state = np.asarray(grip_state, np.float32)
    parts = {"Q1": [kind, rel_target, grip_state], "Q2": [rel_target, grip_state], "Q3": [rel_target]}
    if means is None:
        means, scales = {}, {}
        for name, values in parts.items():
            means[name], scales[name] = [], []
            for part in values:
                m, s = _standardize_fit(part)
                means[name].append(m); scales[name].append(s)
    return ({name: _scaled_query(values, means[name], scales[name]) for name, values in parts.items()},
            means, scales)


def topk_retrieve(memory_q, memory_b, query_q, memory_episode, query_episode,
                  *, topk=TOPK, temperature=SOFTMAX_TEMPERATURE):
    """Cross-episode top-1 and softmax top-k base retrieval.

    Scale the selected-distance spread per query before applying temperature.
    Query variants have different dimensions and native distance scales, so a
    fixed temperature on raw distances would make blend sharpness incomparable.
    """
    memory_q = np.asarray(memory_q, np.float32); query_q = np.asarray(query_q, np.float32)
    memory_b = np.asarray(memory_b, np.float32)
    one = np.zeros((len(query_q), memory_b.shape[1]), np.float32)
    blend = np.zeros_like(one); indices = np.full((len(query_q), topk), -1, np.int64)
    for i, q in enumerate(query_q):
        candidates = np.flatnonzero(np.asarray(memory_episode).astype(str) != str(query_episode[i]))
        if not len(candidates):
            raise ValueError("no cross-episode memory candidates")
        distances = np.linalg.norm(memory_q[candidates] - q[None], axis=1)
        order = np.argsort(distances, kind="stable")[:min(topk, len(candidates))]
        chosen = candidates[order]
        selected_distances = distances[order]
        spread = float(selected_distances.max() - selected_distances.min())
        if spread <= 1e-8:
            weights = np.full(len(chosen), 1.0 / len(chosen), np.float32)
        else:
            relative_distances = (selected_distances - selected_distances.min()) / spread
            logits = -relative_distances / max(temperature, 1e-6)
            weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        one[i] = memory_b[chosen[0]]; blend[i] = weights @ memory_b[chosen]
        indices[i, :len(chosen)] = chosen
    return one, blend, indices


class GeometryPredictor(nn.Module):
    def __init__(self, in_dim, n_classes, width=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.reg = nn.Linear(width, 3); self.cls = nn.Linear(width, n_classes)
    def forward(self, x):
        z = self.body(x); return self.reg(z), self.cls(z)


def fit_geometry_predictor(train_x, train_delta, train_kind, *, epochs=150, seed=0, device="cpu",
                           episode_ids=None, batch_size=512, patience=8):
    """Small supervised regressor/classifier fitted only on training-suite rows."""
    train_x = np.asarray(train_x, np.float32); delta = np.asarray(train_delta, np.float32)
    classes, labels = np.unique(train_kind, return_inverse=True)
    from .experiment import _episode_train_validation_indices
    if episode_ids is None:
        raise ValueError("geometry fitting requires episode IDs for validation")
    fit_ix, val_ix = _episode_train_validation_indices({"episode_id": np.asarray(episode_ids)}, seed + 11)
    x_mean, x_scale = _standardize_fit(train_x[fit_ix])
    y_mean, y_scale = _standardize_fit(delta[fit_ix])
    x = torch.as_tensor(_standardize(train_x, x_mean, x_scale), device=device)
    y = torch.as_tensor(_standardize(delta, y_mean, y_scale), device=device)
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
    torch.manual_seed(seed)
    model = GeometryPredictor(x.shape[1], len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    best, stale, best_state, history = float("inf"), 0, None, []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(fit_ix)
        for lo in range(0, len(order), batch_size):
            ix = order[lo:lo + batch_size]
            reg, logits = model(x[ix])
            loss = F.mse_loss(reg, y[ix]) + F.cross_entropy(logits, labels_t[ix])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            reg, _ = model(x[val_ix])
            # Select for displacement, the quantity used by Q4 retrieval.
            valid = float(F.mse_loss(reg, y[val_ix]).cpu())
        history.append({"epoch": epoch + 1, "validation_delta_mse": valid})
        if valid < best - 1e-6:
            best, stale = valid, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    model.fit_info = {"history": history, "epochs": len(history), "best_epoch": best_epoch,
                      "hit_epoch_cap": len(history) == epochs,
                      "train_rows": len(fit_ix), "validation_rows": len(val_ix),
                      "train_episodes": sorted(set(np.asarray(episode_ids)[fit_ix].tolist())),
                      "validation_episodes": sorted(set(np.asarray(episode_ids)[val_ix].tolist()))}
    return model, (x_mean, x_scale, y_mean, y_scale), classes


def predict_geometry(model, norm_params, classes, x, *, device="cpu"):
    xm, xs, ym, ys = norm_params
    xt = torch.as_tensor(_standardize(x, xm, xs), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        delta, logits = model(xt)
    return (delta.cpu().numpy() * ys + ym).astype(np.float32), classes[logits.argmax(-1).cpu().numpy()]


def _rows(rows, mask):
    return {k: v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v for k, v in rows.items()}


def _load_rows_from_cache(cache_dir, dataset_dir):
    """Read either the standard cache manifest or the heldout feature_manifest."""
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        return load_rows(cache_dir, Path(dataset_dir), max_horizon=HORIZON)
    import tempfile
    manifest = json.loads((cache_dir / "feature_manifest.json").read_text())
    with tempfile.TemporaryDirectory(prefix="geo-query-cache-") as tmp:
        Path(tmp, "manifest.json").write_text(json.dumps(manifest))
        return load_rows(Path(tmp), Path(dataset_dir), max_horizon=HORIZON)


def _command_state_audit(rows, cache_dir, lookahead=5):
    """Check that action-sign switches precede corresponding finger opening changes."""
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        manifest_path = Path(cache_dir) / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    paths = {str(ep["episode_id"]): Path(ep["cache"]) for ep in manifest["episodes"]}
    delta_by_sign = {-1: [], 1: []}
    total_switches = 0
    for episode in np.unique(rows["episode_id"]).astype(str):
        with np.load(paths[episode], allow_pickle=False) as data:
            action = np.asarray(data["action"], np.float32)
            state = np.asarray(data.get("env_state", data["proprio"]), np.float32)
        command = np.sign(action[:, 6]).astype(np.int8)
        nonzero = np.flatnonzero(command != 0)
        switches = nonzero[1:][command[nonzero[1:]] != command[nonzero[:-1]]]
        total_switches += len(switches)
        opening = state[:, 6] - state[:, 7]
        for ix in switches:
            if ix + lookahead < len(opening):
                delta_by_sign[int(command[ix])].append(float(opening[ix + lookahead] - opening[ix - 1]))
    close_delta = np.asarray(delta_by_sign[1], np.float32)
    open_delta = np.asarray(delta_by_sign[-1], np.float32)
    return {
        "command_column": 6,
        "command_values": [-1, 1],
        "switches": int(total_switches),
        "opening_width_definition": "env_state[6] - env_state[7] (mirrored finger qpos)",
        "lookahead_frames": int(lookahead),
        "switch_to_opening_response": {
            "command_to_positive_close": {"n": int(len(close_delta)), "median_width_delta": float(np.median(close_delta)) if len(close_delta) else None,
                                           "fraction_width_decreased": float(np.mean(close_delta < 0)) if len(close_delta) else None},
            "command_to_negative_open": {"n": int(len(open_delta)), "median_width_delta": float(np.median(open_delta)) if len(open_delta) else None,
                                          "fraction_width_increased": float(np.mean(open_delta > 0)) if len(open_delta) else None},
        },
        "selection_reason": "The command is a discrete +/-1 open/close directive; qpos differences at stride-5 cached samples hid most changes and required an arbitrary threshold. Full-rate action switches are direct event markers; s supplies the hand position at that event.",
    }


def _m6_unit_metrics(m6, retrieved, gold):
    exact, near = [], []
    for got, target in zip(retrieved, gold):
        got_id, got_distance = m6._nearest(got)
        gold_id, gold_distance = m6._nearest(target)
        exact.append(got_id is not None and got_id == gold_id
                     and got_distance <= m6.epsilon and gold_distance <= m6.epsilon)
        near.append(m6._functional_distance(got, target) <= m6.epsilon)
    return {"unit_exact": float(np.mean(exact)), "unit_near_epsilon": float(np.mean(near))}


def _target_extent_metrics(offsets, retrieved, gold):
    """Summarize retrieval error when the oracle switch falls inside/outside H."""
    offsets = np.asarray(offsets, np.float32)
    retrieved = np.asarray(retrieved, np.float32)
    gold = np.asarray(gold, np.float32)
    result = {}
    for name, mask in (("within_horizon", offsets <= HORIZON), ("beyond_horizon", offsets > HORIZON)):
        errors = np.linalg.norm(retrieved[mask] - gold[mask], axis=1)
        result[name] = {
            "n": int(mask.sum()),
            "mean_b_l2": float(errors.mean()) if len(errors) else None,
            "median_offset_frames": float(np.median(offsets[mask])) if mask.any() else None,
        }
    return result


def _action_errors(m1, head, bases, rows, actions, *, seed):
    """Per-query errors, with identical residual/white noise for every b source."""
    return {str(t0): np.mean((_predict_abstracted(
        m1, head, bases, rows["f"], rows["s"], t0=t0, seed=seed, draws=2)
        - actions[None]) ** 2, axis=(0, 2, 3)) for t0 in T0S}


def _normal_action_errors(flow, normalizers, rows, *, seed, device):
    rng = np.random.default_rng(seed)
    noises = [rng.standard_normal((len(rows["a"]), int(np.prod(rows["a"].shape[1:])))).astype(np.float32)
              for _ in range(2)]
    _, errors = integrate_flow_sources(flow,normalizers,rows,noises,
                {"normal":None},1.0,device=device,return_per_row=True)
    return errors["normal"]


def _paired_episode_interval(reference, candidate, episodes, *, seed=0):
    """Positive means improvement over Q0. Resample episodes, not windows."""
    diff = np.asarray(reference) - np.asarray(candidate)
    groups, codes = np.unique(np.asarray(episodes), return_inverse=True)
    result = {"mean": float(diff.mean()), "n_episodes": len(groups), "ci95": None}
    if len(groups) >= 10:
        counts = np.bincount(codes)
        sums = np.bincount(codes, weights=diff)
        draws = np.random.default_rng(seed).integers(0, len(groups), (2000, len(groups)))
        means = sums[draws].sum(1) / counts[draws].sum(1)
        result["ci95"] = np.quantile(means, [.025, .975]).tolist()
    return result


def run(*, cache_dir=DEFAULT_CACHE, dataset_dir=DEFAULT_DATASET, output_root=DEFAULT_OUTPUT_ROOT,
        device_name="cuda", seed=0, smoke=False, m1_epochs=60, head_epochs=150, query_epochs=150,
        patience=8, splits=("A", "B")):
    started = time.monotonic()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    rows = _load_rows_from_cache(Path(cache_dir), Path(dataset_dir))
    valid = rows["valid_h25"]
    rows = _rows(rows, valid)
    delta, grip, censored, offsets = oracle_geometric_targets(
        rows, cache_dir=Path(cache_dir), dataset_dir=Path(dataset_dir))
    rows["geo_delta"] = delta; rows["geo_grip"] = grip; rows["geo_censored"] = censored
    rows["geo_offset"] = offsets
    rows["geo_distance"] = np.linalg.norm(np.nan_to_num(delta), axis=1).astype(np.float32)
    all_rows = rows
    if smoke:
        # Small but enough episodes for a train/query split and both suites.
        keep = np.zeros(len(rows["a"]), bool)
        for task in (10, 20, 30):
            eps = np.unique(rows["episode_id"][rows["task_index"] == task])[:3]
            keep |= np.isin(rows["episode_id"], eps)
        rows = _rows(rows, keep)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-s{seed}-{uuid.uuid4().hex[:6]}"
    output = Path(output_root) / f"geo-query-{'smoke-' if smoke else ''}{run_id}"
    output.mkdir(parents=True, exist_ok=False)
    valid_offsets = offsets[~censored]
    valid_distances = np.linalg.norm(delta[~censored], axis=1)
    quantiles = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
    summary = {"oracle": {
        "definition": "first +/- command sign switch after query start, searched through the end of the current skill segment; if it is the final segment, through episode end",
        "position": "full-rate env_state[:,0:3] xyz at action switch; relative to start xyz",
        "proprio_gripper": "indices 6:8 are mirrored left/right finger qpos, not a scalar aperture; opening width = s[6] - s[7]",
        "transition": "sign switch in action[:,6], the seventh action dimension; no qpos delta threshold",
        "action_state_audit": _command_state_audit(all_rows, Path(cache_dir)),
        "rows": int(len(censored)), "valid_target_rows": int((~censored).sum()),
        "no_later_switch_rows": int(censored.sum()), "no_later_switch_fraction": float(censored.mean()),
        "no_later_switch_handling": "No endpoint substitute; rows without a later switch are excluded from Q1-Q4 training and all query comparisons.",
        "target_offset_frames_from_query_start": {"quantiles": {str(q): float(np.quantile(valid_offsets, q)) for q in quantiles},
                                                   "mean": float(np.mean(valid_offsets))},
        "target_within_horizon_rows": int(np.sum(valid_offsets <= HORIZON)),
        "target_beyond_horizon_rows": int(np.sum(valid_offsets > HORIZON)),
        "target_distance_from_hand": {"quantiles": {str(q): float(np.quantile(valid_distances, q)) for q in quantiles},
                                      "mean": float(np.mean(valid_distances))},
        "hindsight_upper_bound_only": True,
    }, "settings": {"seed": seed, "smoke": smoke, "m1_epochs_max": m1_epochs,
                      "head_epochs_max": head_epochs, "query_epochs_max": query_epochs,
                      "patience": patience, "t0s": list(T0S), "splits": list(splits)}, "splits": {}}
    for split, (train_tasks, test_tasks) in SPLITS.items():
        if split not in splits:
            continue
        split_started = time.monotonic()
        train_mask = np.isin(rows["task_index"], train_tasks); test_mask = np.isin(rows["task_index"], test_tasks)
        train = _rows(rows, train_mask); test = _rows(rows, test_mask)
        if smoke:
            # Initial-state identity is unavailable; episode separation alone
            # does not guarantee initial-state separation.
            train = _rows(train, ~np.isin(train["episode_id"], np.unique(train["episode_id"])[-1:]))
            test = _rows(test, np.isin(test["episode_id"], np.unique(test["episode_id"])[:1]))
        if len(train["a"]) < 4 or not len(test["a"]):
            continue
        if set(train["episode_id"]) & set(test["episode_id"]):
            raise ValueError("training and evaluation episodes overlap")
        actions = train["a"][:, :HORIZON].copy(); test_actions = test["a"][:, :HORIZON].copy()
        m6 = M6(epochs=m1_epochs, batch_size=128 if smoke else 512,
                min_support=1 if smoke else 10, min_episodes=1 if smoke else 3,
                seed=seed + (1 if split == "A" else 2), device=str(device))
        m6.fit({"actions":actions,"state":train["s"],"feature":train["f"],"key":train["k"],
                "episode_id":train["episode_id"],"init_state_id":train["episode_id"],
                "seed":seed+(1 if split=="A" else 2)})
        m1 = m6.backbone
        btrain, _ = m6.encode(actions, train["s"]); bgold, _ = m6.encode(test_actions, test["s"])
        mids = _decode_residual_draws(m1, btrain, train["f"], train["s"], draws=4, seed=seed+17)
        torch.manual_seed(seed + 20)
        fiii, fiii_fit = _fit_abstracted_regressor(m1, train, actions, btrain, mids, epochs=head_epochs, batch_size=128 if smoke else 512, patience=patience, seed=seed+20, device=device)
        flow, flow_norm, flow_fit = train_flow(train, horizon=HORIZON, epochs=head_epochs, batch_size=128 if smoke else 512, seed=seed+30, device=device, patience=patience)
        # Geometry needs a future switch target. Censored rows remain in the
        # action/M1 and Q0 feature fit, but not in Q1-Q4 memory or evaluation.
        train_geo_mask = ~train["geo_censored"]
        test_geo_mask = ~test["geo_censored"]
        train_geo = _rows(train, train_geo_mask)
        test_geo = _rows(test, test_geo_mask)
        btrain_geo = btrain[train_geo_mask]
        bgold_geo = bgold[test_geo_mask]
        test_actions_geo = test_actions[test_geo_mask]
        if len(train_geo["a"]) < 4 or not len(test_geo["a"]):
            continue
        # Suite-only memory and cross-episode query candidates.
        kind_train = train_geo["skill_class"].astype(np.int64); kind_test = test_geo["skill_class"].astype(np.int64)
        kind_ids = np.unique(kind_train)  # No evaluation-label vocabulary fit.
        geom_parts_train = {"Q1":[_kind_onehot(kind_train,kind_ids), train_geo["geo_delta"], train_geo["geo_grip"]], "Q2":[train_geo["geo_delta"],train_geo["geo_grip"]], "Q3":[train_geo["geo_delta"]]}
        geom_parts_test = {"Q1":[_kind_onehot(kind_test,kind_ids), test_geo["geo_delta"], test_geo["geo_grip"]], "Q2":[test_geo["geo_delta"],test_geo["geo_grip"]], "Q3":[test_geo["geo_delta"]]}
        queries = {}
        for name in geom_parts_train:
            means_scales = [_standardize_fit(x) for x in geom_parts_train[name]]
            queries[name] = (_scaled_query(geom_parts_train[name], [x[0] for x in means_scales], [x[1] for x in means_scales]),
                             _scaled_query(geom_parts_test[name], [x[0] for x in means_scales], [x[1] for x in means_scales]))
        # Existing CLIP-style language-free key+state encoder; trained only on training suites.
        clip_train = dict(train); clip_test = dict(test)
        clip_train["f"] = np.zeros_like(train["f"]); clip_test["f"] = np.zeros_like(test["f"])
        clip_model, clip_fit = fit_dual_encoder(clip_train, btrain, use_subtask=False, nearby_positive=False,
                                                seed=seed, epochs=CLIP_EPOCHS if not smoke else 2,
                                                patience=CLIP_PATIENCE, device=device, allowed_task_indices=set(train_tasks))
        from .clip_retrieval import _query_embedding
        q0_train = _query_embedding(clip_model, clip_fit, clip_train, device=device)[train_geo_mask]
        q0_test = _query_embedding(clip_model, clip_fit, clip_test, device=device)[test_geo_mask]
        # Q4 changes only Q1's query displacement. Stored geometry and its
        # train-fitted scales stay exact, and kind/grip are identical to Q1.
        # A state-only control is necessary to attribute any benefit to k.
        q4_predictions, q4_fits, q4_rmse = {}, {}, {}
        q1_stats = [_standardize_fit(p) for p in geom_parts_train["Q1"]]
        for name, features in (("Q4", ("k", "s")), ("Q4_state", ("s",))):
            model, norm_params, classes = fit_geometry_predictor(
                np.concatenate([train_geo[k] for k in features], axis=1),
                train_geo["geo_delta"], kind_train, epochs=query_epochs,
                seed=seed, device=device, episode_ids=train_geo["episode_id"],
                batch_size=128 if smoke else 512, patience=patience)
            predicted, _ = predict_geometry(model, norm_params, classes,
                np.concatenate([test_geo[k] for k in features], axis=1), device=device)
            q4_predictions[name] = _scaled_query(
                [_kind_onehot(kind_test, kind_ids), predicted, test_geo["geo_grip"]],
                [x[0] for x in q1_stats], [x[1] for x in q1_stats])
            q4_fits[name] = model.fit_info
            q4_rmse[name] = float(np.sqrt(np.mean((predicted-test_geo["geo_delta"])**2)))
        q_by_name={
            "Q0":(q0_train,q0_test,btrain_geo,train_geo["episode_id"],test_geo["episode_id"],bgold_geo),
            **{n:(queries[n][0],queries[n][1],btrain_geo,train_geo["episode_id"],test_geo["episode_id"],bgold_geo) for n in queries},
            **{name:(queries["Q1"][0], prediction,btrain_geo,train_geo["episode_id"],test_geo["episode_id"],bgold_geo)
               for name,prediction in q4_predictions.items()},
        }
        split_results = {"n_memory_all":len(train["a"]),"n_query_all":len(test["a"]),
                         "n_memory_with_target":len(train_geo["a"]),"n_query_with_target":len(test_geo["a"]),
                         "m1_epochs":m1.epochs_run,"flow_fit":flow_fit,"fiii_fit":fiii_fit,
                         "clip_fit":{k:v for k,v in clip_fit.items() if k != "normalizers"},
                         "q4_fit":q4_fits,"retrieval":{}, "q4_delta_rmse":q4_rmse}
        # Existing skill-class labels are retained as an analysis stratum.
        known = np.isin(kind_test, np.unique(kind_train))
        strata={"trained_kind":known,"untrained_kind":~known}
        # Controls and random sources are fixed once per full query set. All
        # b sources see the same noise; H/label strata only slice these errors.
        rng = np.random.default_rng(seed + 71)
        random_indices = np.asarray([rng.choice(np.flatnonzero(train_geo["episode_id"] != ep))
                                     for ep in test_geo["episode_id"]])
        controls = {
            "gold_b_iii_mse": _action_errors(m1, fiii, bgold_geo, test_geo, test_actions_geo, seed=seed+43),
            "random_b_iii_mse": _action_errors(m1, fiii, btrain_geo[random_indices], test_geo, test_actions_geo, seed=seed+43),
            "shuffled_b_iii_mse": _action_errors(m1, fiii, bgold_geo[rng.permutation(len(bgold_geo))], test_geo, test_actions_geo, seed=seed+43),
        }
        normal = _normal_action_errors(flow,flow_norm,test_geo,seed=seed+51,device=device)
        audit = {"query_episode":test_geo["episode_id"], "query_frame":test_geo["frame"],
                 "memory_episode":train_geo["episode_id"], "memory_frame":train_geo["frame"],
                 "target_offset":test_geo["geo_offset"], "random_indices":random_indices}
        reference_errors = {}
        for qname,(qm,qt,memory_b,memory_episodes,query_episodes,query_gold) in q_by_name.items():
            one,blend,ix=topk_retrieve(qm,memory_b,qt,memory_episodes,query_episodes)
            audit[f"{qname}_indices"] = ix
            vals={}
            for strategy,bhat in (("top1",one),("softmax_topk",blend)):
                b_errors = np.linalg.norm(bhat-query_gold,axis=1)
                action_errors = _action_errors(m1,fiii,bhat,test_geo,test_actions_geo,seed=seed+43)
                audit[f"{qname}_{strategy}_b_l2"] = b_errors
                for t0, errors in action_errors.items():
                    audit[f"{qname}_{strategy}_mse_{t0}"] = errors
                if qname == "Q0":
                    reference_errors[strategy] = (b_errors, action_errors)
                vals[strategy]={}
                for stratum,mask in strata.items():
                    if not mask.any(): continue
                    metrics={"n":int(mask.sum()),"base_l2":float(b_errors[mask].mean()),**_m6_unit_metrics(m6,bhat[mask],query_gold[mask])}
                    metrics["target_extent_b_l2"] = _target_extent_metrics(
                        test_geo["geo_offset"][mask], bhat[mask], query_gold[mask]
                    )
                    metrics["normal_flow_mse"] = float(normal[mask].mean())
                    for name, errors in {"iii_action_mse":action_errors,**controls}.items():
                        metrics[name] = {t:float(e[mask].mean()) for t,e in errors.items()}
                    for extent, extent_mask in (("within_horizon",test_geo["geo_offset"]<=HORIZON),
                                                 ("beyond_horizon",test_geo["geo_offset"]>HORIZON)):
                        selected = mask & extent_mask
                        em = metrics["target_extent_b_l2"][extent]
                        for name, errors in {"iii_action_mse":action_errors,**controls}.items():
                            em[name] = {t:float(e[selected].mean()) if selected.any() else None for t,e in errors.items()}
                    ref_b, ref_a = reference_errors[strategy]
                    metrics["paired_improvement_over_Q0"] = {
                        "b_l2": _paired_episode_interval(ref_b[mask],b_errors[mask],query_episodes[mask],seed=seed),
                        "iii_mse": {t:_paired_episode_interval(ref_a[t][mask],e[mask],query_episodes[mask],seed=seed)
                                    for t,e in action_errors.items()}}
                    vals[strategy][stratum]=metrics
            split_results["retrieval"][qname]=vals
        split_results["elapsed_seconds"] = time.monotonic() - split_started
        summary["splits"][split]=split_results
        np.savez_compressed(
            output / f"split-{split}.npz",
            b_memory=btrain_geo,
            b_gold=bgold_geo,
            delta_train=train_geo["geo_delta"],
            delta_query=test_geo["geo_delta"],
            **audit,
        )
    summary["elapsed_seconds"] = time.monotonic() - started
    (output/"results.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    (output/"report.md").write_text(_report(summary))
    return output,summary


def _report(result):
    oracle = result["oracle"]
    lines = [
        "# Geometric query smoke/study", "",
        "## Oracle definition and coverage", "",
        oracle["definition"] + ". This is a hindsight upper-bound diagnostic and is unavailable at inference time.",
        "", f"- Eligible intervals: {oracle['valid_target_rows']:,}/{oracle['rows']:,}; no later switch: {oracle['no_later_switch_rows']:,} ({oracle['no_later_switch_fraction']:.1%}).",
        f"- Gripper: {oracle['proprio_gripper']}. Event: {oracle['transition']}.",
        "- No later switch rows have no substitute target and are excluded from every query's evaluation and memory candidate pool. Q1–Q4 also exclude them when fitting geometry; Q0's query encoder may use them for its training-suite-only fit.",
        "- Target offset from query start, frames: " + ", ".join(f"p{q}={v:.1f}" for q, v in oracle["target_offset_frames_from_query_start"]["quantiles"].items()) + f"; mean={oracle['target_offset_frames_from_query_start']['mean']:.1f}. Within H=25: {oracle['target_within_horizon_rows']:,}; beyond H=25: {oracle['target_beyond_horizon_rows']:,}.",
        "- Hand-to-target distance: " + ", ".join(f"p{q}={v:.3f} m" for q, v in oracle["target_distance_from_hand"]["quantiles"].items()) + f"; mean={oracle['target_distance_from_hand']['mean']:.3f} m.",
        "- Command/state check: " + json.dumps(oracle["action_state_audit"]["switch_to_opening_response"], ensure_ascii=False),
        "", "## Retrieval results", "",
        ("Smoke uses small episode subsets and short training runs; no performance conclusion." if result["settings"]["smoke"] else
         "Full-row study. Check seed stability and the fit diagnostics before interpreting a null result."),
        "Trained-kind and untrained-kind query strata are reported separately when present.",
    ]
    for split, split_result in result["splits"].items():
        lines.extend(["",f"### Split {split}","",
                      f"Memory {split_result['n_memory_with_target']}/{split_result['n_memory_all']}; query {split_result['n_query_with_target']}/{split_result['n_query_all']}.",
                      "", "| Split | Query | Retrieval (stratum) | N | b L2 | M6 exact | M6 within ε | F_iii MSE t0=.2 | F MSE (10 steps) | Gold b F_iii MSE | b L2 within H | b L2 beyond H |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for query, strategies in split_result["retrieval"].items():
            for strategy, strata in strategies.items():
                for stratum, m in strata.items():
                    extent = m["target_extent_b_l2"]
                    within, beyond = extent["within_horizon"], extent["beyond_horizon"]
                    within_text = f"{within['mean_b_l2']:.4f} (n={within['n']})" if within["mean_b_l2"] is not None else f"— (n={within['n']})"
                    beyond_text = f"{beyond['mean_b_l2']:.4f} (n={beyond['n']})" if beyond["mean_b_l2"] is not None else f"— (n={beyond['n']})"
                    lines.append(f"| {split} | {query} | {strategy} ({stratum}) | {m['n']} | {m['base_l2']:.4f} | {m['unit_exact']:.3f} | {m['unit_near_epsilon']:.3f} | {m['iii_action_mse']['0.2']:.4f} | {m['normal_flow_mse']:.4f} | {m['gold_b_iii_mse']['0.2']:.4f} | {within_text} | {beyond_text} |")
        lines.extend(["", f"### Split {split}: target prediction and training", "",
                      f"- Displacement RMSE (m): {split_result['q4_delta_rmse']}.",
                      f"- Epochs: M1 {split_result['m1_epochs']}/{result['settings']['m1_epochs_max']}; F_iii {len(split_result['fiii_fit']['history'])}/{result['settings']['head_epochs_max']} (best {split_result['fiii_fit']['best_epoch']}); Q0 {split_result['clip_fit']['epochs']}; Q4: " + str({k:(v['epochs'],v['best_epoch']) for k,v in split_result['q4_fit'].items()}),
                      "- Reaching an epoch cap while validation still improves requires a longer run before claiming no benefit. If gold b does not beat random b, the head has not demonstrated sensitivity to retrieval.",
                      "", "| Query | Strategy / kind | Target | N | F_iii .2 | Gold b .2 | Random b .2 |",
                      "|---|---|---|---:|---:|---:|---:|"])
        for query, strategies in split_result["retrieval"].items():
            for strategy, strata in strategies.items():
                for stratum, m in strata.items():
                    for extent, e in m["target_extent_b_l2"].items():
                        if e['n']:
                            lines.append(f"| {query} | {strategy} / {stratum} | {extent} | {e['n']} | {e['iii_action_mse']['0.2']:.4f} | {e['gold_b_iii_mse']['0.2']:.4f} | {e['random_b_iii_mse']['0.2']:.4f} |")
        lines.extend(["", "| Query | Strategy / kind | Δ MSE versus Q0 (.2; positive is better) | Episode bootstrap 95% interval |",
                      "|---|---|---:|---|"])
        for query, strategies in split_result['retrieval'].items():
            for strategy, strata in strategies.items():
                for stratum, m in strata.items():
                    p = m['paired_improvement_over_Q0']['iii_mse']['0.2']
                    lines.append(f"| {query} | {strategy} / {stratum} | {p['mean']:.5f} | {p['ci95']} ({p['n_episodes']} episodes) |")
    lines.extend([
        "", "Q0 uses the existing symmetric CLIP-style g·g retrieval with language-bearing `f` zeroed, so the query is k+s. Q1 is action-kind + target displacement + opening width; Q2 removes action-kind; Q3 uses displacement only. Q4 changes ONLY Q1's query displacement to the k+s prediction; Q4_state uses s alone. Both retain oracle kind and exact stored geometry/scales. Compare Q4 against Q4_state to isolate information contributed by the mean image feature. The auxiliary kind prediction is not used for retrieval. Q1's action-kind is an oracle label, not a language embedding: Q1–Q2 isolates the label contribution and Q2–Q3 isolates gripper state. Query standardization is fitted on training suites. Search candidates exclude the query episode. Softmax top-k uses k=5; selected distances are scaled to [0,1] within each query before temperature 0.25. Top-1 provides the comparison without blend-weight effects.",
        "", "Both b and action errors are separated at target offset ≤25 versus >25. Within-H hindsight supplies part of the outcome; do not treat improvement there as deployable retrieval. Beyond-H still uses future information and is only a diagnostic. No future target enters F_iii directly. Normal F starts from pure noise at t=1 and integrates 10 steps; it never receives the gold action as its input. F_iii uses the same residual and white-noise draws for every b source, and MSE averages errors over draws, not predictions.",
        "", "## Smoke judgment and limitations", "",
        "No automatic hypothesis verdict. CIs resample paired episode clusters conditional on the fitted model/split; fewer than 10 episodes gives no CI. These are marginal exploratory intervals, not simultaneous significance tests. Check seeds 0–2 and both splits. Raw b L2 is comparable only within a fitted M1; use action MSE or paired within-model differences across seeds. Initial-state IDs are absent: different episodes/suites do not prove initial-state separation, and no claim of initial-state generalization is supported. Coverage is conditional on a later switch; censored intervals are not evaluated. A weak Q4 result does not prove information is absent from k: this is one predictor, and k+s without instruction can be ambiguous even when the objects are visible.",
        "", "Full run: `for seed in 0 1 2; do python -m skill_moe.skillnet.examples.emergent.light.intermediate.geo_query --device cuda --seed \"$seed\" --splits A B --m1-epochs 60 --head-epochs 150 --query-epochs 150; done` (after activating /work/env/activate.sh).", "",
    ])
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--dataset-dir",type=Path,default=DEFAULT_DATASET)
    parser.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device",default="cuda"); parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--splits",nargs="+",choices=tuple(SPLITS),default=["A","B"])
    parser.add_argument("--m1-epochs",type=int,default=60)
    parser.add_argument("--head-epochs",type=int,default=150)
    parser.add_argument("--query-epochs",type=int,default=150)
    parser.add_argument("--patience",type=int,default=8)
    args=parser.parse_args()
    path,result=run(cache_dir=args.cache_dir,dataset_dir=args.dataset_dir,output_root=args.output_root,device_name=args.device,seed=args.seed,smoke=args.smoke,m1_epochs=5 if args.smoke else args.m1_epochs,head_epochs=4 if args.smoke else args.head_epochs,query_epochs=4 if args.smoke else args.query_epochs,patience=args.patience,splits=args.splits)
    print(json.dumps({"output":str(path),"splits":{k:{"n_memory_with_target":v["n_memory_with_target"],"n_query_with_target":v["n_query_with_target"],"queries":list(v["retrieval"])} for k,v in result["splits"].items()}},indent=2))

if __name__ == "__main__": main()
