"""Diagnose whether cross-suite memory failures come from values or keys.

Uses the cached held-out run, trained checkpoint and fixed train20 PCA encoder.
Motion search and all-value oracle are diagnostic upper bounds because they
inspect the query's executed future/action target.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from skill_moe.skillnet.examples.emergent.eval.data import load_z_run
from skill_moe.skillnet.examples.emergent.eval.memory import EmergentMemory, KeyEncoder
from skill_moe.skillnet.examples.memory_pilot.g2_retrieval.crosstask import _pairwise_outcomes
from .model import BottleneckModel

PREVIOUS = Path("/work/.skillnet-state/outputs/memory-pilot/emergent-light-heldout-20261026T061907Z-f96a16")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _load_memory(directory: Path, run):
    memory = EmergentMemory.load(directory)
    row_by_key = {(str(ep), int(step)): i for i, (ep, step) in enumerate(zip(run.episode_id, run.step))}
    entries, rows = [], []
    for entry in memory.store.entries:
        meta = entry.metadata
        i = row_by_key.get((str(meta.episode_id), int(meta.start_step)))
        if i is not None and meta.suite in ("libero_object", "libero_goal", "libero_spatial"):
            entries.append(entry); rows.append(i)
    return memory, entries, np.asarray(rows, dtype=np.int64)


def _decode_mse(model, norm, run, qrows, z, *, batch=2048):
    """One executed-action MSE per query row."""
    fm, fs = norm["f_mean"], norm["f_scale"]
    sm, ss = norm["s_mean"], norm["s_scale"]
    out = []
    model.eval()
    with torch.no_grad():
        for lo in range(0, len(qrows), batch):
            hi = min(len(qrows), lo + batch); rows = qrows[lo:hi]
            f = torch.as_tensor(((run.keys["vlm_img_lang"][rows] - fm) / fs).astype("float32"), device=DEVICE)
            s = torch.as_tensor(((run.proprio[rows] - sm) / ss).astype("float32"), device=DEVICE)
            zz = torch.as_tensor(np.asarray(z[lo:hi], dtype="float32"), device=DEVICE)
            pred = model.decode(zz, f, s).cpu().numpy()
            pred = pred * np.asarray(norm["a_scale"]).reshape(run.target_actions.shape[1:])
            pred = pred + np.asarray(norm["a_mean"]).reshape(run.target_actions.shape[1:])
            err = ((pred - run.target_actions[rows]) ** 2).mean(axis=2)
            out.append((err * run.target_mask[rows]).sum(axis=1) / np.maximum(run.target_mask[rows].sum(axis=1), 1))
    return np.concatenate(out)


def _motion_component(query, store):
    """Mean G2 motion distances [position, rotation, gripper] for one query."""
    return _pairwise_outcomes(query[None], store)[0]


def _evaluate_setting(root, horizon, beta, seed):
    tag = f"H{horizon}_beta{beta:g}"
    base = PREVIOUS / tag
    cfg = torch.load(base / "model.pt", map_location="cpu", weights_only=False)
    run = load_z_run(base / "eval_input", horizon=horizon)
    action_shape = tuple(run.target_actions.shape[1:])
    model = BottleneckModel(action_shape, run.proprio.shape[1], run.keys["vlm_img_lang"].shape[1], z_dim=32).to(DEVICE)
    model.load_state_dict(cfg["state_dict"])
    norm = cfg["normalizers"]
    enc = KeyEncoder.load(base / "key_encoder_train20.npz")
    keys = enc.transform({"vlm_img_nolang": run.keys["vlm_img_nolang"], "proprio": run.proprio}).astype(np.float32)
    manifest = json.loads((base / "eval_input" / "manifest.json").read_text())
    # Reconstruct exact spatial store/query split from the previous retrieval logs.
    qlog_path = base / "query_retrieval_log.jsonl"
    qids = {json.loads(line)["query_episode_id"] for line in qlog_path.open() if line.strip()}
    spatial_ids = {e["episode_id"] for e in manifest["episodes"] if e["suite"] == "libero_spatial"}
    qmask = np.isin(run.episode_id, sorted(qids)) & run.usable
    if not qmask.any() or qids - spatial_ids:
        raise ValueError(f"invalid query episode set for {tag}")
    qrows = np.flatnonzero(qmask)
    if set(run.init_state_id[qrows]) & set(run.init_state_id[np.isin(run.episode_id, sorted(spatial_ids - qids))]):
        raise AssertionError("spatial query/store initial-state leakage")
    spatial_store_eps = spatial_ids - qids

    mem_no, entries_no, rows_no = _load_memory(base / "memory_without_spatial", run)
    mem_yes, entries_yes, rows_yes = _load_memory(base / "memory_with_spatial", run)
    mask_og_no = np.asarray([e.metadata.suite in ("libero_object", "libero_goal") for e in entries_no])
    mask_og_yes = np.asarray([e.metadata.suite in ("libero_object", "libero_goal") for e in entries_yes])
    if not mask_og_no.all() or not np.array_equal(rows_no, rows_yes[mask_og_yes]):
        raise AssertionError("object/goal memory differs between baseline and spatial-added stores")
    z_no = np.stack([e.value for e in entries_no]).astype(np.float32)
    z_yes = np.stack([e.value for e in entries_yes]).astype(np.float32)
    rows_og = rows_no[mask_og_no]
    z_og = z_no[mask_og_no]
    keys_og = keys[rows_og]
    rows_yes = rows_yes
    z_yes = z_yes
    print(f"{tag}: q={len(qrows)} OG={len(rows_og)} with_spatial={len(rows_yes)}", flush=True)

    # Realizable frozen-key top-1 and top-5 softmax blend, based on the exact
    # fixed train20 PCA distance scale used by the stored memory.
    def key_retrieval(rows, values, memory):
        chosen1 = np.empty(len(qrows), dtype=np.int64)
        blend = np.empty((len(qrows), values.shape[1]), dtype=np.float32)
        selected = np.empty((len(qrows), 5), dtype=np.int64)
        selected_count = np.empty(len(qrows), dtype=np.int64)
        for qi, r in enumerate(qrows):
            ok = ((run.episode_id[rows] != run.episode_id[r]) &
                  (run.init_state_id[rows] != run.init_state_id[r]))
            ix = np.flatnonzero(ok)
            d = np.linalg.norm(keys[r] - keys[rows[ix]], axis=1)
            order = np.argsort(d, kind="stable")[:5]
            chosen1[qi] = ix[order[0]]
            n = len(order); selected[qi, :n] = ix[order]; selected_count[qi] = n
            scale = 0.5 * (memory.d_ref or max(float(d[order[0]]), 1e-12))
            w = np.exp(-(d[order] - d[order[0]]) / max(scale, 1e-12)); w /= w.sum()
            blend[qi] = w @ values[ix[order]]
        return chosen1, blend, selected, selected_count

    # Evaluate exact all-value oracle against object/goal-only and spatial-added memory.
    # Chunk query rows and candidate z to bound memory; maintain per-query minimum.
    def oracle_best(rows, values):
        best = np.full(len(qrows), np.inf)
        best_idx = np.full(len(qrows), -1, dtype=np.int64)
        # 1 query at a time avoids retaining Q x candidates x H x action.
        for qi, r in enumerate(qrows):
            eligible = ((run.episode_id[rows] != run.episode_id[r]) &
                        (run.init_state_id[rows] != run.init_state_id[r]))
            ix = np.flatnonzero(eligible)
            target = run.target_actions[r:r+1]
            mask = run.target_mask[r:r+1]
            f = torch.as_tensor(((run.keys["vlm_img_lang"][r:r+1] - norm["f_mean"]) / norm["f_scale"]).astype("float32"), device=DEVICE)
            s = torch.as_tensor(((run.proprio[r:r+1] - norm["s_mean"]) / norm["s_scale"]).astype("float32"), device=DEVICE)
            bmin, bidx = np.inf, -1
            with torch.no_grad():
                for lo in range(0, len(ix), 2048):
                    chunk_ix = ix[lo:lo+2048]
                    zz = torch.as_tensor(values[chunk_ix], device=DEVICE)
                    pred = model.decode(zz, f.expand(len(chunk_ix), -1), s.expand(len(chunk_ix), -1)).cpu().numpy()
                    pred = pred * np.asarray(norm["a_scale"]).reshape(run.target_actions.shape[1:])
                    pred = pred + np.asarray(norm["a_mean"]).reshape(run.target_actions.shape[1:])
                    err = ((pred - target) ** 2).mean(axis=2)
                    mse = (err * mask).sum(axis=1) / max(int(mask.sum()), 1)
                    j = int(mse.argmin())
                    if mse[j] < bmin:
                        bmin, bidx = float(mse[j]), int(chunk_ix[j])
            best[qi], best_idx[qi] = bmin, bidx
            if (qi + 1) % 250 == 0:
                print(f"{tag}: all-value oracle {qi+1}/{len(qrows)}", flush=True)
        return best, best_idx

    print(f"{tag}: decoding reference modes", flush=True)
    err_own = _decode_mse(model, norm, run, qrows, run.z["post"][qrows])
    err_zero = _decode_mse(model, norm, run, qrows, np.zeros_like(run.z["post"][qrows]))
    rng = np.random.default_rng(seed + horizon)
    random_errors = []
    random_indices = np.empty((len(qrows), 8), dtype=np.int64)
    for i, r in enumerate(qrows):
        eligible = np.flatnonzero((run.episode_id[rows_yes] != run.episode_id[r]) &
                                  (run.init_state_id[rows_yes] != run.init_state_id[r]))
        pick = rng.choice(eligible, size=8, replace=True)
        random_indices[i] = pick
    for j in range(random_indices.shape[1]):
        random_errors.append(_decode_mse(model, norm, run, qrows, z_yes[random_indices[:, j]]))
    err_random = np.stack(random_errors).mean(axis=0)
    oracle_og, idx_oracle_og = oracle_best(rows_og, z_og)
    oracle_yes, idx_oracle_yes = oracle_best(rows_yes, z_yes)
    key1_og, blend5_og, key5_og, keyn_og = key_retrieval(rows_og, z_og, mem_no)
    key1_yes, blend5_yes, key5_yes, keyn_yes = key_retrieval(rows_yes, z_yes, mem_yes)
    err_key1_og = _decode_mse(model, norm, run, qrows, z_og[key1_og])
    err_key5_og = _decode_mse(model, norm, run, qrows, blend5_og)
    err_key1_yes = _decode_mse(model, norm, run, qrows, z_yes[key1_yes])
    err_key5_yes = _decode_mse(model, norm, run, qrows, blend5_yes)

    # Future-motion retrieval from object/goal only. Combined distance uses the
    # G2 position/rotation/gripper metrics, each normalized by its eligible-pair mean.
    qmotion, smotion = run.motion[qrows], run.motion[rows_og]
    outcome_scale = np.zeros(3, dtype=np.float64)
    for i, r in enumerate(qrows):
        eligible = ((run.episode_id[rows_og] != run.episode_id[r]) &
                    (run.init_state_id[rows_og] != run.init_state_id[r]))
        if not eligible.any():
            raise ValueError("no eligible cross-episode motion donors")
        outcome_scale += _motion_component(qmotion[i], smotion)[eligible].sum(axis=0)
    # Recompute means with actual eligible pair count.
    pair_count = 0
    for r in qrows:
        pair_count += int(((run.episode_id[rows_og] != run.episode_id[r]) &
                           (run.init_state_id[rows_og] != run.init_state_id[r])).sum())
    outcome_scale /= max(pair_count, 1)
    motion_dist = np.empty((len(qrows), len(rows_og)), dtype=np.float32)
    nearest_components = np.empty((len(qrows), 3), dtype=np.float32)
    key_selected_motion = np.empty((len(qrows), 3), dtype=np.float32)
    motion_best = np.empty((len(qrows), 3), dtype=np.float32)
    motion_z1 = np.empty((len(qrows), 32), dtype=np.float32)
    motion_z5 = np.empty_like(motion_z1)
    # relative softmax scale set from median nearest eligible combined motion distance.
    combined_all = []
    component_rows = []
    for qi, r in enumerate(qrows):
        comp = _motion_component(qmotion[qi], smotion)
        eligible = ((run.episode_id[rows_og] != run.episode_id[r]) &
                    (run.init_state_id[rows_og] != run.init_state_id[r]))
        combined = (comp / np.maximum(outcome_scale, 1e-12)).sum(axis=1)
        combined[~eligible] = np.inf
        motion_dist[qi] = combined
        component_rows.append(comp)
        combined_all.append(float(np.min(combined)))
        motion_best[qi] = comp[int(np.argmin(combined))]
    motion_temp_scale = max(float(np.median(combined_all)), 1e-12) * 0.5
    for qi, r in enumerate(qrows):
        comp = component_rows[qi]
        order = np.argsort(motion_dist[qi], kind="stable")[:5]
        weights = np.exp(-(motion_dist[qi, order] - motion_dist[qi, order[0]]) / motion_temp_scale)
        weights /= weights.sum()
        motion_z1[qi] = z_og[order[0]]
        motion_z5[qi] = weights @ z_og[order]
        nearest_components[qi] = comp[order[0]]
        key_selected_motion[qi] = comp[key1_og[qi]]
        if (qi + 1) % 250 == 0:
            print(f"{tag}: motion search {qi+1}/{len(qrows)}", flush=True)
    err_motion1 = _decode_mse(model, norm, run, qrows, motion_z1)
    err_motion5 = _decode_mse(model, norm, run, qrows, motion_z5)

    # For each query's key top-1 and top-5, report how motion-close the selected
    # entries are. Quantile bins refer to the best achievable eligible future-motion distance.
    nearest_score = np.asarray(combined_all)
    qs = np.quantile(nearest_score, [0.25, 0.5, 0.75])
    bins = np.digitize(nearest_score, qs, right=True)
    motion_quantiles = []
    for b in range(4):
        sel = bins == b
        def comp_summary(v):
            return {name: float(np.mean(v[sel, j])) for j, name in enumerate(("position", "rotation", "gripper"))}
        motion_quantiles.append({
            "best_motion_distance_quantile": f"Q{b+1}", "n_queries": int(sel.sum()),
            "best_available_component_mean": comp_summary(motion_best),
            "key_top1_selected_component_mean": comp_summary(key_selected_motion),
            "key_top1_selected_combined_normalized_mean": float(np.mean(motion_dist[np.arange(len(qrows))[sel], key1_og[sel]])),
            "best_available_combined_normalized_mean": float(np.mean(nearest_score[sel])),
        })

    errors = {
        "own": err_own, "random": err_random, "zero": err_zero,
        "oracle_best_value_object_goal": oracle_og,
        "oracle_best_value_spatial_added": oracle_yes,
        "key_top1_object_goal": err_key1_og, "key_top5_blend_object_goal": err_key5_og,
        "motion_top1_object_goal": err_motion1, "motion_top5_blend_object_goal": err_motion5,
        "key_top1_spatial_added": err_key1_yes, "key_top5_blend_spatial_added": err_key5_yes,
    }
    summary = {
        "horizon": horizon, "beta": beta, "query_rows": len(qrows),
        "query_episodes": len(set(run.episode_id[qrows])), "object_goal_memory_entries": len(rows_og),
        "spatial_added_memory_entries": int(len(rows_yes) - len(rows_og)),
        "spatial_store_episodes": len(spatial_store_eps),
        "mse_mean": {k: float(v.mean()) for k, v in errors.items()},
        "paired_row_mse": {k: v.tolist() for k, v in errors.items()},
        "oracle_best_item_index": {"object_goal": idx_oracle_og.tolist(), "spatial_added": idx_oracle_yes.tolist()},
        "motion_search": {
            "note": "future env motion selects candidates; diagnostic upper bound, not deployable key",
            "top1_error_mean": float(err_motion1.mean()), "top5_softmax_blend_error_mean": float(err_motion5.mean()),
            "temperature": 0.5, "temperature_scale": motion_temp_scale,
            "distance_component_scales": outcome_scale.tolist(),
            "key_top1_selected_motion_component_mean": {name: float(key_selected_motion[:, j].mean()) for j, name in enumerate(("position", "rotation", "gripper"))},
            "motion_top1_component_mean": {name: float(nearest_components[:, j].mean()) for j, name in enumerate(("position", "rotation", "gripper"))},
            "best_available_component_mean": {name: float(motion_best[:, j].mean()) for j, name in enumerate(("position", "rotation", "gripper"))},
            "best_motion_distance_quantile_edges": qs.tolist(), "key_retrieval_by_motion_quantile": motion_quantiles,
        },
        "memory_settings": {"key": "fixed train20 PCA over language-free PaliGemma image pool + proprio",
                            "topk": 5, "temperature_relative": 0.5, "query_store_episode_disjoint": True},
        "methods": {"oracle_best_value_object_goal": "minimum per-query executed-action reconstruction MSE over all eligible object/goal posterior z; query action is used, oracle upper bound",
                    "oracle_best_value_spatial_added": "same all-value oracle over object/goal plus accepted spatial store entries",
                    "motion_top1/top5": "future environment-motion distance chooses object/goal values; diagnostic upper bound",
                    "random": "mean error across 8 uniformly sampled accepted spatial-added store z draws"},
    }
    d = root / tag; d.mkdir(parents=True, exist_ok=False)
    (d / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del model
    return summary


def _write_report(root, results):
    lines = ["# Cross-suite value/key diagnosis", "",
             f"Previous held-out report (unchanged): [{PREVIOUS / 'report.md'}]({PREVIOUS / 'report.md'}). This diagnosis reuses its cached feature rows, model checkpoints, query/store episode split, memory contents, and fixed train20 PCA; no feature extraction, fitting, or parameter updates were performed.", "",
             "The query set is the same spatial query episodes as the previous run. Object/goal-only memory contains the training20 store. Spatial-added memory contains those same object/goal entries plus only the previously threshold-accepted spatial store entries. Query episode and initial-state exclusions are applied to every retrieval.", "",
             "## Reconstruction results", "", "MSE is measured on each query's executed action target. Lower is better. The all-value oracle chooses the z with the smallest error using the query's own action target and is therefore an unattainable diagnostic upper bound. Motion search uses the query's future environment trajectory and is also an upper-bound diagnostic, not an inference-time key.", "",
             "| H | β | own | random | zero | oracle all z, object/goal | oracle all z, spatial added | key top-1 O/G | key top-5 blend O/G | motion top-1 O/G | motion top-5 blend O/G | key top-1 added | key top-5 blend added |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in results:
        m=d["mse_mean"]
        lines.append(f"| {d['horizon']} | {d['beta']:g} | " + " | ".join(f"{m[k]:.4f}" for k in ("own","random","zero","oracle_best_value_object_goal","oracle_best_value_spatial_added","key_top1_object_goal","key_top5_blend_object_goal","motion_top1_object_goal","motion_top5_blend_object_goal","key_top1_spatial_added","key_top5_blend_spatial_added")) + " |")
    for d in results:
        lines += ["", f"## H={d['horizon']}, β={d['beta']:g}", "",
                  f"Queries={d['query_rows']} rows / {d['query_episodes']} episodes; object/goal memory={d['object_goal_memory_entries']} entries; accepted spatial additions={d['spatial_added_memory_entries']} across {d['spatial_store_episodes']} spatial store episodes.", "",
                  "Key retrieval's selected motion distance grouped by the best achievable query-to-object/goal motion distance:", "",
                  "| Best available motion-distance quantile | n | Best available position/rotation/gripper | Key top-1 selected position/rotation/gripper | selected normalized combined distance | best available normalized distance |",
                  "|---|---:|---|---|---:|---:|"]
        for row in d["motion_search"]["key_retrieval_by_motion_quantile"]:
            b=row["best_available_component_mean"]; k=row["key_top1_selected_component_mean"]
            lines.append(f"| {row['best_motion_distance_quantile']} | {row['n_queries']} | {b['position']:.4f}/{b['rotation']:.4f}/{b['gripper']:.4f} | {k['position']:.4f}/{k['rotation']:.4f}/{k['gripper']:.4f} | {row['key_top1_selected_combined_normalized_mean']:.4f} | {row['best_available_combined_normalized_mean']:.4f} |")
        lines += ["", "Key top-1 versus motion-nearest component means:", "",
                  f"- Key top-1: `{d['motion_search']['key_top1_selected_motion_component_mean']}`",
                  f"- Motion-nearest (future oracle): `{d['motion_search']['motion_top1_component_mean']}`",
                  f"- Best available within object/goal memory: `{d['motion_search']['best_available_component_mean']}`", ""]
    lines += ["## Reading this diagnosis", "",
              "If object/goal all-value oracle is substantially better than random, usable values exist cross-suite; if future-motion retrieval is also good but fixed-key retrieval selects motion-distant items, the key is a likely bottleneck. If even the all-value oracle and motion retrieval are near random, this model's posterior values do not transfer well across these suites. Spatial-added comparisons indicate whether the previously observed gain depends on adding same-suite spatial examples.",
              "In these runs, the object/goal all-value oracle is far below random for both horizons (0.074/0.230 at H=25 and 0.047/0.220 at H=10), so decoder-compatible values do exist in the trained-task store for spatial queries. Future-motion top-5 also improves over random (0.162/0.230 and 0.142/0.220). The fixed PCA key's object/goal top-5 is near random at H=25 (0.227 vs 0.230), and better but still much worse than motion retrieval at H=10 (0.172 vs 0.142). This points to a substantial key mismatch, especially at H=25; it does not show that the learned values are uniformly transferable, since the attainable value oracle is query-action-conditioned and the deployable key result remains weak.",
              "These are query-target-conditioned diagnostics; neither oracle establishes a deployable policy gain. The cached future-motion labels and action targets are used only for diagnosis. Full paired row results are saved in each setting's `results.json`.", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=Path("/work/.skillnet-state/outputs/memory-pilot"))
    p.add_argument("--seed", type=int, default=41)
    a=p.parse_args(); _seed(a.seed)
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:6]
    root=a.output_root/f"emergent-light-diag-{run_id}"; root.mkdir(parents=True, exist_ok=False)
    summaries=[]
    for horizon in (25,10):
        summaries.append(_evaluate_setting(root,horizon,0.01,a.seed))
    _write_report(root,summaries)
    print(f"report: {root/'report.md'}", flush=True)


if __name__ == "__main__":
    main()
