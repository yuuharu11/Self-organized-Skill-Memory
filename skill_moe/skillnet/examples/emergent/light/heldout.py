"""Evaluate memory-added spatial demos for tasks excluded from z-model training."""
from __future__ import annotations

import argparse
import json
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from skill_moe.skillnet.examples.emergent.eval.crossrecon import cross_reconstruction, _masked_mse
from skill_moe.skillnet.examples.emergent.eval.data import load_z_run
from skill_moe.skillnet.examples.emergent.eval.gonogo import GoNoGoCriteria, decide
from skill_moe.skillnet.examples.emergent.eval.memory import EmergentMemory, KeyEncoder, MemoryConfig
from skill_moe.skillnet.examples.emergent.eval.metrics import MIConfig, evaluate_mi
from skill_moe.skillnet.examples.memory_pilot.g2_retrieval.crosstask import (
    METRICS, _cluster_bootstrap, _knn_weights, _pairwise_outcomes, _standardized_distances,
)
from .extract import extract_cache
from .model import BottleneckModel
from .run import DEFAULT_CKPT, DEFAULT_DATA, DEFAULT_OUTPUTS, _collect, _fit, _write_eval_input

PREVIOUS_RUN = Path("/work/.skillnet-state/outputs/memory-pilot/emergent-light-20261026T054800Z-85a58f")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _rows_for(records, episodes):
    ids = {e["episode_id"] for e in episodes}
    rows = []
    for ep in records:
        if ep["episode_id"] not in ids:
            continue
        d = ep["cache_arrays"]
        for j, action in enumerate(ep["windows"]):
            if action is None:
                continue
            rows.append({"ep": ep, "j": j, "step": int(d["step"][j]),
                         "f": d["f"][j].astype(np.float32), "k": d["k"][j].astype(np.float32),
                         "s": d["proprio"][j].astype(np.float32), "a": action.astype(np.float32)})
    return rows


def _load_cached_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _split_spatial(episodes, seed: int):
    by_task: dict[str, list] = {}
    for ep in episodes:
        by_task.setdefault(ep["task"], []).append(ep)
    rng = np.random.default_rng(seed)
    query, store = [], []
    for task, eps in sorted(by_task.items()):
        perm = rng.permutation(len(eps))
        n_query = max(1, int(round(len(eps) * 0.25)))
        n_query = min(n_query, len(eps) - 1)
        q = set(perm[:n_query].tolist())
        for i, ep in enumerate(eps):
            (query if i in q else store).append(ep)
    qids = {e["episode_id"] for e in query}
    sids = {e["episode_id"] for e in store}
    if qids & sids:
        raise AssertionError("spatial episode leakage")
    return query, store


def _fixed_calibration(memory: EmergentMemory) -> tuple[float | None, float | None, int]:
    """Chunked equivalent of EmergentMemory.calibrate; avoids an O(N²) distance matrix."""
    keys = memory._matrix()
    n = len(keys)
    if n < 2:
        memory.tau = memory.d_ref = None
        return None, None, 0
    episode = np.asarray([e.metadata.episode_id for e in memory.store.entries], dtype=object)
    init_state = np.asarray([e.metadata.init_state_id for e in memory.store.entries], dtype=object)
    key_sq = np.sum(keys * keys, axis=1)
    nearest = np.full(n, np.inf, dtype=np.float64)
    for start in range(0, n, 128):
        stop = min(n, start + 128)
        d2 = np.maximum(key_sq[start:stop, None] + key_sq[None] - 2 * keys[start:stop] @ keys.T, 0.0)
        invalid = episode[start:stop, None] == episode[None]
        invalid |= init_state[start:stop, None] == init_state[None]
        d2[invalid] = np.inf
        nearest[start:stop] = np.sqrt(d2.min(axis=1))
    finite = nearest[np.isfinite(nearest)]
    if len(finite) < memory.config.min_entries_for_threshold:
        memory.tau = memory.d_ref = None
        return None, None, len(finite)
    tau = float(np.quantile(finite, memory.config.threshold_quantile))
    d_ref = float(np.median(finite))
    memory.tau, memory.d_ref = tau, d_ref
    return tau, d_ref, len(finite)


def _seed_memory(memory: EmergentMemory, run, rows: np.ndarray, keys: np.ndarray) -> None:
    for i in rows:
        memory.add(keys[i], run.z["post"][i], episode_id=str(run.episode_id[i]),
                   init_state_id=str(run.init_state_id[i]), task=str(run.task[i]), suite=str(run.suite[i]),
                   step=int(run.step[i]), interval_len=run.horizon)


def _retrieval_ii(run, query_rows: np.ndarray, store_rows: np.ndarray, keys: np.ndarray,
                  *, bootstrap: int, seed: int) -> dict[str, Any]:
    qmotion, smotion = run.motion[query_rows], run.motion[store_rows]
    eligible = ((run.episode_id[query_rows, None] != run.episode_id[None, store_rows])
                & (run.init_state_id[query_rows, None] != run.init_state_id[None, store_rows])
                & (run.task[query_rows, None] != run.task[None, store_rows]))
    if not eligible.any(axis=1).all():
        raise ValueError("a spatial query row has no other-task store candidate")
    outcome = _pairwise_outcomes(qmotion, smotion)
    random_row = np.empty((len(query_rows), len(METRICS)))
    oracle = {1: np.empty_like(random_row), 5: np.empty_like(random_row)}
    for i in range(len(query_rows)):
        idx = np.flatnonzero(eligible[i])
        vals = outcome[i, idx]
        random_row[i] = vals.mean(axis=0)
        for k in oracle:
            ke = min(k, len(idx))
            oracle[k][i] = np.sort(vals, axis=0)[:ke].mean(axis=0)
    features = {"z_prior": run.z["prior"][query_rows], "key_pca_nolang": keys[query_rows]}
    store_features = {"z_prior": run.z["prior"][store_rows], "key_pca_nolang": keys[store_rows]}
    by_method = {}
    for name, qfeat in features.items():
        dist = _standardized_distances(qfeat, store_features[name])
        values = {1: np.empty_like(random_row), 5: np.empty_like(random_row)}
        for i in range(len(query_rows)):
            idx = np.flatnonzero(eligible[i])
            for k in values:
                w, ke = _knn_weights(dist[i, idx], k)
                values[k][i] = w @ outcome[i, idx] / ke
        by_method[name] = {}
        for k, pred in values.items():
            random_gap = random_row - oracle[k]
            closed = np.divide(random_row - pred, random_gap, out=np.full_like(pred, np.nan),
                               where=np.abs(random_gap) > 1e-12)
            by_method[name][str(k)] = {
                "retrieved_mean": dict(zip(METRICS, pred.mean(axis=0).tolist())),
                "random_mean": dict(zip(METRICS, random_row.mean(axis=0).tolist())),
                "oracle_mean": dict(zip(METRICS, oracle[k].mean(axis=0).tolist())),
                "gap_closed": dict(zip(METRICS, np.nanmean(closed, axis=0).tolist())),
            }
    # Whole initial-state bootstrap for the paired random-minus-retrieval differences.
    diff_cols = []
    labels = []
    for name in by_method:
        dist = _standardized_distances(features[name], store_features[name])
        for k in (1, 5):
            pred = np.empty_like(random_row)
            for i in range(len(query_rows)):
                idx = np.flatnonzero(eligible[i]); w, ke = _knn_weights(dist[i, idx], k)
                pred[i] = w @ outcome[i, idx] / ke
            diff_cols.append(random_row - pred)
            labels.extend({"method": name, "k": k, "metric": m} for m in METRICS)
    diffs = np.concatenate(diff_cols, axis=1)
    boot = _cluster_bootstrap(diffs, run.init_state_id[query_rows], seed, bootstrap)
    intervals = [{"method": x["method"], "k": x["k"], "metric": x["metric"],
                  "random_minus_method": e, "ci95_familywise": ci}
                 for x, e, ci in zip(labels, boot["estimates"], boot["intervals"])]
    return {"n_query_rows": int(len(query_rows)), "n_query_init_states": int(len(np.unique(run.init_state_id[query_rows]))),
            "n_store_rows": int(len(store_rows)), "eligible_other_task_candidates_mean": float(eligible.sum(1).mean()),
            "methods": by_method, "paired_random_improvement": intervals,
            "bootstrap": {k: boot[k] for k in ("n_groups", "n_comparisons", "bootstrap_samples")}}


def _infer_retrieved(memory: EmergentMemory, query_rows: np.ndarray, keys: np.ndarray, run):
    blend = np.full_like(run.z["prior"], np.nan, dtype=np.float32)
    top1 = np.full_like(blend, np.nan)
    logs = []
    entry_by_id = {e.metadata.entry_id: e for e in memory.store.entries}
    for i in query_rows:
        r = memory.retrieve(keys[i], episode_id=str(run.episode_id[i]),
                            init_state_id=str(run.init_state_id[i]), count_usage=False)
        if r.z is None:
            continue
        blend[i] = r.z
        top1[i] = entry_by_id[r.candidate_ids[0]].value
        logs.append({"query_episode_id": str(run.episode_id[i]), "init_state_id": str(run.init_state_id[i]),
                     "step": int(run.step[i]), "candidate_ids": list(r.candidate_ids),
                     "weights": r.weights.tolist(), "distances": r.distances.tolist(),
                     "excluded_same_episode_or_init_state": True})
    return blend, top1, logs


def _evaluate_one(out_root: Path, tag: str, horizon: int, beta: float, records, train_eps, spatial_eps,
                  train_rows, spatial_rows, spatial_query_eps, spatial_store_eps, *, seed: int, epochs: int,
                  train_batch: int, bootstrap: int, permutations: int,
                  resume_existing: bool = False) -> dict[str, Any]:
    all_episodes = train_eps + spatial_eps
    infer_rows = train_rows + spatial_rows
    config_dir = out_root / tag
    model_path = config_dir / "model.pt"
    eval_input = config_dir / "eval_input"
    run = None
    if resume_existing:
        required = [model_path, config_dir / "key_encoder_train20.npz",
                    config_dir / "memory_with_spatial/store.npz", config_dir / "memory_without_spatial/store.npz",
                    eval_input / "manifest.json", config_dir / "spatial_add_log.jsonl"]
        if any(not p.exists() for p in required) or (config_dir / "results.json").exists():
            raise ValueError(f"cannot resume incomplete or already completed configuration {config_dir}")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        norm, history = checkpoint["normalizers"], checkpoint["history"]
        arrays = _load_cached_npz(eval_input / "episodes" / f"{all_episodes[0]['episode_id']}.npz")
        action_shape = arrays["z_target_actions"].shape[1:]
        run = load_z_run(eval_input, horizon=horizon)
        model = BottleneckModel(action_shape, run.proprio.shape[1],
                                run.keys["vlm_img_lang"].shape[1], z_dim=32).to(DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        config_dir.mkdir(exist_ok=False)
        model, preds, norm, history = _fit(train_rows, infer_rows, horizon, beta, seed, epochs=epochs,
                                            batch_size=train_batch, device=DEVICE)
        torch.save({"state_dict": model.cpu().state_dict(), "normalizers": norm, "history": history}, model_path)
    train_ids = {e["episode_id"] for e in train_eps}
    info = {"checkpoint": str(model_path), "seed": seed, "beta": beta, "epochs": epochs,
            "training_episodes": sorted(train_ids)}
    if not resume_existing:
        _write_eval_input(eval_input, all_episodes, infer_rows, preds, horizon, info)
    if run is None:
        run = load_z_run(eval_input, horizon=horizon)
    print(f"{tag}: model fit and eval input ready", flush=True)
    qids = {e["episode_id"] for e in spatial_query_eps}
    sids = {e["episode_id"] for e in spatial_store_eps}
    spatial_all_ids = {e["episode_id"] for e in spatial_eps}
    if qids & sids or (qids | sids) != spatial_all_ids:
        raise AssertionError("spatial query/store split is not episode-disjoint and exhaustive")
    spatial_query = np.isin(run.episode_id, sorted(qids)) & run.usable
    spatial_store = np.isin(run.episode_id, sorted(sids)) & run.usable
    train_mask = np.isin(run.episode_id, sorted(train_ids)) & run.usable
    if set(run.init_state_id[spatial_query]) & set(run.init_state_id[spatial_store]):
        raise AssertionError("spatial initial-state leakage")
    if not spatial_query.any() or not spatial_store.any() or not train_mask.any():
        raise ValueError("empty train, spatial store, or spatial query set")

    # PCA is fit only on object+goal (the 20 trained tasks); spatial rows enter transform only.
    if resume_existing:
        encoder = KeyEncoder.load(config_dir / "key_encoder_train20.npz")
    else:
        encoder = KeyEncoder.fit({"vlm_img_nolang": run.keys["vlm_img_nolang"][train_mask],
                                  "proprio": run.proprio[train_mask]}, include_language=False,
                                 n_components=32)
        encoder.save(config_dir / "key_encoder_train20.npz")
    key_inputs = {"vlm_img_nolang": run.keys["vlm_img_nolang"], "proprio": run.proprio}
    keys = encoder.transform(key_inputs)

    memory_config = MemoryConfig(k=5, temperature=0.5, threshold_quantile=0.9, threshold_mode="fixed",
                                  min_entries_for_threshold=10, exclude_same_episode=True,
                                  exclude_same_init_state=True, require_full_interval=True)
    add_order = np.flatnonzero(spatial_store)
    add_order = add_order[np.lexsort((run.step[add_order], run.episode_id[add_order]))]
    add_dir = config_dir / "spatial_add_log.jsonl"
    if resume_existing:
        with_spatial = EmergentMemory.load(config_dir / "memory_with_spatial")
        no_spatial = EmergentMemory.load(config_dir / "memory_without_spatial")
        tau, d_ref = no_spatial.tau, no_spatial.d_ref
        calibration_n = len(no_spatial.store)
        add_reasons = {}
        for line in add_dir.open(encoding="utf-8"):
            reason = json.loads(line)["reason"]
            add_reasons[reason] = add_reasons.get(reason, 0) + 1
        accepted_keys = {(e.metadata.episode_id, e.metadata.start_step) for e in with_spatial.store.entries
                         if e.metadata.suite == "libero_spatial"}
        accepted_spatial = np.array([(str(run.episode_id[i]), int(run.step[i])) in accepted_keys
                                     for i in range(len(run))], dtype=bool)
    else:
        with_spatial = EmergentMemory(encoder, memory_config, horizon=horizon, log_path=add_dir)
        no_spatial = EmergentMemory(encoder, memory_config, horizon=horizon)
        _seed_memory(with_spatial, run, np.flatnonzero(train_mask), keys)
        _seed_memory(no_spatial, run, np.flatnonzero(train_mask), keys)
        tau, d_ref, calibration_n = _fixed_calibration(no_spatial)
        with_spatial.tau, with_spatial.d_ref = tau, d_ref
        add_reasons = {}
        accepted_spatial = np.zeros(len(run), dtype=bool)
        for i in add_order:
            _, entry = with_spatial.observe(keys[i], run.z["post"][i], executed=bool(run.executed[i]),
                                             interval_len=int(run.interval_len[i]),
                                             episode_id=str(run.episode_id[i]), init_state_id=str(run.init_state_id[i]),
                                             task=str(run.task[i]), suite=str(run.suite[i]), step=int(run.step[i]))
            reason = with_spatial.log[-1]["reason"]
            add_reasons[reason] = add_reasons.get(reason, 0) + 1
            accepted_spatial[i] = entry is not None
    additions = sum(v for k, v in add_reasons.items() if k.startswith("added"))
    add_rate = additions / len(add_order) if len(add_order) else 0.0
    if not resume_existing:
        with_spatial.save(config_dir / "memory_with_spatial")
        no_spatial.save(config_dir / "memory_without_spatial")
    print(f"{tag}: spatial additions {additions}/{len(add_order)}", flush=True)

    z_add, z_add_top1, query_log = _infer_retrieved(with_spatial, np.flatnonzero(spatial_query), keys, run)
    z_base, z_base_top1, base_log = _infer_retrieved(no_spatial, np.flatnonzero(spatial_query), keys, run)
    for name, zmat in (("memory_added", z_add), ("memory_no_spatial", z_base),
                       ("memory_top1_added", z_add_top1), ("memory_top1_no_spatial", z_base_top1)):
        # Store-side cluster centres use the actual posterior values stored in memory;
        # query-side rows use the retrieved values.
        value = np.array(run.z["post"], copy=True)
        value[spatial_query] = zmat[spatial_query]
        run.z[name] = value
    if not resume_existing:
        (config_dir / "query_retrieval_log.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in query_log), encoding="utf-8")
        (config_dir / "query_retrieval_no_spatial_log.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in base_log), encoding="utf-8")

    mi = evaluate_mi(run, spatial_query, train_mask,
                     MIConfig(cluster_counts=(8, 16, 32), primary_clusters=16, n_permutations=permutations,
                              bootstrap_samples=bootstrap, seed=seed, minimum_groups=10),
                     sources=("prior", "post", "untrained_prior", "memory_added", "memory_no_spatial",
                              "memory_top1_added"))
    criterion_i = decide(mi, None, None, GoNoGoCriteria())["criteria"]["i_prior_mi"]
    print(f"{tag}: motion MI complete ({criterion_i['status']})", flush=True)

    store_rows = np.flatnonzero(train_mask | accepted_spatial)
    q_rows = np.flatnonzero(spatial_query)
    ii = _retrieval_ii(run, q_rows, store_rows, keys, bootstrap=bootstrap, seed=seed)
    print(f"{tag}: other-task retrieval complete", flush=True)

    fm, fs, sm, ss = norm["f_mean"], norm["f_scale"], norm["s_mean"], norm["s_scale"]
    def reconstruct(z, context):
        ix = context["row_index"]
        f = torch.from_numpy(((run.keys["vlm_img_lang"][ix] - fm) / fs).astype(np.float32)).to(DEVICE)
        s = torch.from_numpy(((context["proprio"] - sm) / ss).astype(np.float32)).to(DEVICE)
        zz = torch.from_numpy(np.asarray(z, dtype=np.float32)).to(DEVICE)
        with torch.no_grad():
            pred = model.to(DEVICE).eval().decode(zz, f, s).cpu().numpy().reshape(len(ix), -1)
        return (pred * norm["a_scale"] + norm["a_mean"]).reshape(len(ix), horizon, -1)

    qmask = np.zeros(len(run), dtype=bool); qmask[q_rows] = True
    smask = np.zeros(len(run), dtype=bool); smask[store_rows] = True
    cr = cross_reconstruction(run, reconstruct, qmask, smask, key=keys,
                              extra_z={"memory_blend_added": z_add, "memory_blend_no_spatial": z_base,
                                       "memory_top1_added": z_add_top1},
                              seed=seed, random_draws=8, bootstrap_samples=bootstrap)
    print(f"{tag}: cross-reconstruction complete", flush=True)
    # Direct paired add-vs-no-add CI over the same spatial query rows.
    base_err, add_err = {}, {}
    target, mask = run.target_actions[q_rows], run.target_mask[q_rows]
    for label, zmat, dest in (("without", z_base, base_err), ("with", z_add, add_err)):
        predictions = []
        for start in range(0, len(q_rows), 256):
            rows = q_rows[start:start + 256]
            ctx = {"row_index": rows, "proprio": run.proprio[rows]}
            predictions.append(reconstruct(zmat[rows], ctx))
        dest["row_mse"] = _masked_mse(np.concatenate(predictions), target, mask)
    add_vs_base = base_err["row_mse"] - add_err["row_mse"]
    paired = _cluster_bootstrap(add_vs_base[:, None], run.init_state_id[q_rows], seed + 21, bootstrap)
    criterion_iii = next((x for x in cr["improvement_over_prior"] if x["mode"] == "memory_blend_added"), None)

    summary = {"tag": tag, "horizon": horizon, "beta": beta, "n_train_tasks": 20,
               "train_suites": ["libero_object", "libero_goal"], "n_train_episodes": len(train_eps),
               "n_train_rows": len(train_rows), "n_spatial_episodes": len(spatial_eps),
               "n_spatial_store_episodes": len(spatial_store_eps), "n_spatial_query_episodes": len(spatial_query_eps),
               "n_spatial_store_rows": int(spatial_store.sum()), "n_spatial_query_rows": int(spatial_query.sum()),
               "excluded_tail_windows_train": sum(sum(w is None for w in x["windows"]) for x in train_eps),
               "threshold": {"quantile": memory_config.threshold_quantile, "tau": tau, "d_ref": d_ref,
                             "calibration_entries": calibration_n, "add_candidates": int(len(add_order)),
                             "additions": int(additions), "add_rate": float(add_rate), "reasons": add_reasons},
               "memory_sizes": {"without_spatial": len(no_spatial.store), "with_spatial": len(with_spatial.store)},
               "mi": mi, "criterion_i": criterion_i, "retrieval_ii": ii,
               "cross_reconstruction": cr,
               "iii_prime": {"estimate_prior_minus_added_blend": (criterion_iii["estimate"] if criterion_iii else None),
                             "ci95_familywise": (criterion_iii["ci95_familywise"] if criterion_iii else None),
                             "status": ("pass" if criterion_iii and criterion_iii["ci95_familywise"] and
                                        criterion_iii["ci95_familywise"][0] > 0 else
                                        "fail" if criterion_iii and criterion_iii["estimate"] <= 0 else "undetermined")},
               "addition_effect_added_minus_no_spatial": {"estimate_no_spatial_minus_added": float(add_vs_base.mean()),
                                                          "ci95_familywise": paired["intervals"][0],
                                                          "bootstrap": paired["bootstrap_samples"]},
               "latent_diagnostics": {"train_objective_kl_last_epoch": history[-1]["kl_nats"],
                                      "train_recon_mse_last_epoch": history[-1]["recon_mse"],
                                      "heldout_post_to_prior_kl": mi["posterior"].get("kl_post_to_prior_nats_per_row"),
                                      "heldout_post_kl_to_standard_normal": mi["posterior"]["kl_to_standard_normal_nats_per_row"],
                                      "effective_dims_kl_gt_0.01": mi["posterior"]["active_dims_kl_gt_0.01"],
                                      "z_dim": 32},
               "method_notes": {"memory_top1": "nearest stored posterior z under fixed train20 PCA key",
                                "memory_blend": "top-5 softmax(-distance / (0.5 * train-memory d_ref))",
                                "criterion_iii_prime": "familywise paired 95% cluster bootstrap; positive prior-minus-memory means improvement",
                                "crosstask_ii": "computed only on spatial query rows, eligible store candidates from different tasks",
                                "spatial_split": "whole episodes stratified by task; query/store IDs disjoint"}}
    (config_dir / "results.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    _write_config_report(summary, config_dir / "report.md")
    del model
    return summary


def _write_config_report(d: dict[str, Any], path: Path):
    c = d["criterion_i"]
    iii = d["iii_prime"]
    lines = [f"# Held-out task evaluation {d['tag']}", "",
             f"H={d['horizon']}, β={d['beta']}; train tasks=20 (object/goal), train episodes={d['n_train_episodes']}, "
             f"rows={d['n_train_rows']}; spatial store/query episodes={d['n_spatial_store_episodes']}/{d['n_spatial_query_episodes']} "
             f"and usable windows={d['n_spatial_store_rows']}/{d['n_spatial_query_rows']}.", "",
             f"PCA/reference fit rows: object+goal only. Added spatial entries: {d['threshold']['additions']}/{d['threshold']['add_candidates']} "
             f"({d['threshold']['add_rate']:.1%}); tau={d['threshold']['tau']}, d_ref={d['threshold']['d_ref']}.", "",
             "| Measure | Result |", "|---|---:|",
             f"| i prior-motion MI status | {c['status']} |",
             f"| Prior env/joint MI shuffle-adjusted bits | {c.get('motion_shuffle_adjusted_bits', float('nan')):.4f} |",
             f"| Prior minus untrained MI bits | {c.get('minus_untrained_bits', float('nan')):.4f} |",
             f"| iii' prior minus memory-added MSE | {iii['estimate_prior_minus_added_blend']} |",
             f"| iii' familywise 95% CI | {iii['ci95_familywise']} |",
             f"| iii' status | {iii['status']} |",
             f"| add-vs-no-spatial MSE (no-spatial minus added) | {d['addition_effect_added_minus_no_spatial']['estimate_no_spatial_minus_added']:.6g} |",
             f"| add-vs-no-spatial familywise 95% CI | {d['addition_effect_added_minus_no_spatial']['ci95_familywise']} |", "",
             "Cross-reconstruction MSE modes:", "",
             "| Mode | MSE |", "|---|---:|"]
    for mode, mse in d["cross_reconstruction"]["mse"].items():
        lines.append(f"| {mode} | {mse:.6g} |")
    lines += ["", "Other-task retrieval gap closed (spatial query only):", "",
              "| Feature | k | position | rotation | gripper |", "|---|---:|---:|---:|---:|"]
    for feature, by_k in d["retrieval_ii"]["methods"].items():
        for k, values in by_k.items():
            gap = values["gap_closed"]
            lines.append(f"| {feature} | {k} | {gap['eef_relative_position']:.3f} | "
                         f"{gap['eef_relative_rotation_rad']:.3f} | {gap['gripper_relative']:.3f} |")
    lines += ["", f"Objective train KL(q‖p), last epoch: {d['latent_diagnostics']['train_objective_kl_last_epoch']:.5g}; "
              f"held-out KL(q‖p): {d['latent_diagnostics']['heldout_post_to_prior_kl']:.5g}; "
              f"effective dims: {d['latent_diagnostics']['effective_dims_kl_gt_0.01']}/32.", "",
              "Query/store split is by whole spatial episodes within each task. The model and normalizers use only object/goal; fixed PCA is fit only to object/goal rows. Spatial store is streamed through the fixed novelty threshold with no parameter updates. The prior run's same-task iii result remains unchanged.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(root: Path, results: list[dict[str, Any]], cache_manifest: dict[str, Any], args):
    lines = ["# Held-out task memory evaluation", "",
             f"Prior report (unchanged): [{PREVIOUS_RUN / 'report.md'}]({PREVIOUS_RUN / 'report.md'}). Its criterion iii was evaluated on other episodes from seen tasks; this run tests spatial queries for a Prior trained only on object/goal.",
             "", f"Cache reused: `{args.cache_dir}`; no VLM features re-extracted. Total cached episodes={len(cache_manifest['episodes'])}; "
             f"sampled feature rows={sum(e['n_rows'] for e in cache_manifest['episodes'])}.",
             "Training: libero_object + libero_goal only (20 tasks); spatial is excluded from action/state/feature normalization, model training, and PCA fit. H=25 and H=10; β=0.001 and 0.01; z_dim=32; objective normalized action MSE + β·analytic KL(q‖p).",
             "Spatial episodes are split within task by whole episode (75% memory store / 25% query, seeded). Memory starts with all executed training20 posterior z/key rows, then spatial store rows stream through the fixed train20 PCA and fixed q=0.9 novelty threshold. No model parameters change during memory insertion.",
             "", "| H | β | spatial add rate | i prior MI | ii prior gap position | ii key gap position | iii' prior−added MSE (95% CI) | add effect no-spatial−added (95% CI) | iii' |",
             "|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for d in results:
        gap_prior = d["retrieval_ii"]["methods"]["z_prior"]["1"]["gap_closed"]["eef_relative_position"]
        gap_key = d["retrieval_ii"]["methods"]["key_pca_nolang"]["1"]["gap_closed"]["eef_relative_position"]
        iii = d["iii_prime"]
        ae = d["addition_effect_added_minus_no_spatial"]
        ci = iii["ci95_familywise"]
        add_ci = ae["ci95_familywise"]
        lines.append(f"| {d['horizon']} | {d['beta']:g} | {d['threshold']['add_rate']:.1%} | {d['criterion_i']['status']} | "
                     f"{gap_prior:.3f} | {gap_key:.3f} | {iii['estimate_prior_minus_added_blend']:.6g} "
                     f"({ci}) | {ae['estimate_no_spatial_minus_added']:.6g} ({add_ci}) | {iii['status']} |")
    lines += ["", "## Interpretation and limits", "",
              "Criterion iii' passes only if the familywise paired 95% interval for Prior MSE minus memory-added blend MSE is strictly above zero. The add-effect column compares the no-spatial memory and the same model with threshold-accepted spatial entries; positive no-spatial-minus-added favors spatial insertion.",
              "The main prior result remains the original all-task report and is not edited. In this held-out setting, the decoder is still the small MLP action decoder; this does not establish benefit in a VLA policy.",
              "G2 task-cross retrieval uses observed environment motion as the evaluation target. No LIBERO skill or annotation columns are read. Subtask labels remain synthetic evaluator proxies and are not evidence for this held-out criterion.",
              "Per-setting numerical evidence and artifacts are in `H*_beta*/report.md` and `results.json`.", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    if args.resume_run is not None:
        out_root = args.resume_run
        if not out_root.is_dir() or (out_root / "report.md").exists():
            raise ValueError("--resume-run must point to an incomplete heldout output directory")
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        out_root = args.output_root / f"emergent-light-heldout-{run_id}"
        out_root.mkdir(parents=True, exist_ok=False)
    cache_manifest = extract_cache(args.dataset, args.cache_dir, args.checkpoint, stride=5,
                                   batch_size=args.vision_batch)
    feature_manifest_path = out_root / "feature_manifest.json"
    if feature_manifest_path.exists():
        existing = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        if existing.get("dataset") != cache_manifest.get("dataset") or len(existing.get("episodes", [])) != len(cache_manifest["episodes"]):
            raise ValueError("heldout output cache manifest does not match the requested feature cache")
    else:
        with feature_manifest_path.open("x", encoding="utf-8") as f:
            json.dump(cache_manifest, f, indent=2)
    summaries = []
    for horizon in (25, 10):
        records, _ = _collect(cache_manifest, horizon)
        from .extract import SUITE_BY_TASK
        for record in records:
            record["suite"] = SUITE_BY_TASK.get(int(record["task_index"]), record["suite"])
        train_eps = [e for e in records if e["suite"] in ("libero_object", "libero_goal")]
        spatial_eps = [e for e in records if e["suite"] == "libero_spatial"]
        if len({e["task_index"] for e in train_eps}) != 20 or len({e["task_index"] for e in spatial_eps}) != 10:
            raise ValueError("cache does not contain exactly object+goal 20 train tasks and 10 spatial held-out tasks")
        spatial_query_eps, spatial_store_eps = _split_spatial(spatial_eps, args.seed)
        train_rows = _rows_for(records, train_eps)
        spatial_rows = _rows_for(records, spatial_eps)
        if not train_rows or not spatial_rows:
            raise ValueError(f"empty usable train or spatial set for H={horizon}")
        for beta in (1e-3, 1e-2):
            tag = f"H{horizon}_beta{beta:g}"
            summary = _evaluate_one(out_root, tag, horizon, beta, records, train_eps, spatial_eps,
                                    train_rows, spatial_rows, spatial_query_eps, spatial_store_eps,
                                    seed=args.seed, epochs=args.epochs, train_batch=args.train_batch,
                                    bootstrap=args.bootstrap, permutations=args.permutations,
                                    resume_existing=(out_root / tag).exists())
            summaries.append(summary)
            print(f"completed {tag}: iii_prime={summary['iii_prime']['status']} add_rate={summary['threshold']['add_rate']:.3f}",
                  flush=True)
    _write_report(out_root, summaries, cache_manifest, args)
    print(out_root, flush=True)
    return out_root


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--cache-dir", type=Path,
                   default=Path("/work/.skillnet-state/outputs/memory-pilot/emergent-light-20261026T050453Z-0cb536/feature-cache"))
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUTS)
    p.add_argument("--resume-run", type=Path, default=None)
    p.add_argument("--vision-batch", type=int, default=2)
    p.add_argument("--train-batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--permutations", type=int, default=100)
    p.add_argument("--bootstrap", type=int, default=500)
    run(p.parse_args())


if __name__ == "__main__":
    main()
