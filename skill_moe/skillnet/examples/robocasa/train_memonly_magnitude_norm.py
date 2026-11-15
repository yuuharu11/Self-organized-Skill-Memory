#!/usr/bin/env python3
"""Train a history-free scale head with the selected-unit position norm ratio."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MP = HERE.parent / "memory_pilot" / "robocasa"
sys.path.insert(0, str(MP))
import experiment as ex
import magnitude_head as mh


def norm_ratio_label(pool, unit_id, task, episode, frame, episode_actions):
    """log(||truth_L|| / ||unit_L||), using EEF translation in stored order."""
    lo, hi = map(int, pool["action_offsets"][unit_id:unit_id + 2])
    unit = np.asarray(pool["actions_flat"][lo:hi], np.float32)
    truth = np.asarray(episode_actions[(int(task), int(episode))], np.float32)
    n = min(len(unit), max(0, len(truth) - int(frame)))
    if n < 1:
        return 1.0
    # LeRobot source order puts EEF translation at action[5:8]; reordering to
    # policy order leaves these three channels as fork action[:3].
    den = float(np.linalg.norm(unit[:n, 5:8]))
    num = float(np.linalg.norm(truth[int(frame):int(frame) + n, 5:8]))
    if den < 1e-10 or num < 1e-10:
        return 1.0
    return float(np.clip(num / den, 0.25, 4.0))


def main():
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=mh.KEYLITE / "magnitude_v2")
    p.add_argument("--max-epochs", type=int, default=150)
    args = p.parse_args()
    out = args.output
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output {out}")
    out.mkdir(parents=True)
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "8")))
    seed = 20261112

    rk = mh._load_keylite()
    z = np.load(mh.CTX / "cache/rows.npz", allow_pickle=False)
    pool = dict(np.load(mh.CTX / "cache/pool.npz", allow_pickle=False))
    feat = np.asarray(np.load(mh.CTX / "cache/prefix_mean_fp16.npy", mmap_mode="r"), np.float32)
    ck = torch.load(mh.KEYLITE / "context_history.pt", map_location="cpu", weights_only=False)
    if not ck.get("input_history", False):
        raise AssertionError("Context+history retrieval checkpoint is required")
    manifest = json.loads((mh.SOURCE / "manifest.json").read_text())
    episode_actions = {
        (int(d["task_id"]), int(d["episode"])):
            ex._load_cached_episode(mh.SOURCE, int(d["task_id"]), int(d["episode"]))[0]
        for d in manifest["demos"]
    }

    # Same context/history ranker and same memory-side candidate selection as
    # the earlier magnitude experiment; only target and magnitude features differ.
    mem = mh._make_rows(rk, ck, z, pool, feat, "memory", episode_actions)
    held = mh._make_rows(rk, ck, z, pool, feat, "heldout", episode_actions)
    uk, _ = mh._unit_metadata(pool)
    qmem, _ = rk.assemble_context(z, feat, ck["norm"], "memory")
    qheld, _ = rk.assemble_context(z, feat, ck["norm"], "heldout")
    x = np.concatenate((np.asarray(qmem, np.float32), uk[mem["pick"]]), axis=1).astype(np.float32)
    xheld = np.concatenate((np.asarray(qheld, np.float32), uk[held["pick"]]), axis=1).astype(np.float32)
    y = np.asarray([
        norm_ratio_label(pool, int(j), int(z["memory_task"][i]),
                         int(z["memory_episode"][i]) % 1000, int(z["memory_frame"][i]), episode_actions)
        for i, j in enumerate(mem["pick"])
    ], np.float32)

    mem_eps = np.asarray(z["memory_episode"], np.int64)
    held_eps = np.asarray(z["heldout_episode"], np.int64)
    dev_eps = set(map(int, ck["dev_episodes"]))
    val_mask = np.asarray([int(e) in dev_eps for e in mem_eps])
    fit_ids, val_ids = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
    if np.intersect1d(mem_eps[fit_ids], held_eps).size:
        raise AssertionError("heldout episodes leaked into magnitude-head training")
    model, train = mh._fit(x, y, fit_ids, val_ids, seed=seed, max_epochs=args.max_epochs,
                           label="history_free_position_norm_ratio")
    ckpt = {
        "state": train["state"], "feature_mean": train["feature_mean"],
        "feature_std": train["feature_std"], "best_epoch": train["best_epoch"],
        "val_logscale_mse": train["val_logscale_mse"], "input_dim": int(x.shape[1]),
        "history_free": True, "target": "log clipped(||truth_L_pos|| / ||selected_unit_L_pos||)",
        "fit_rows": len(fit_ids), "validation_rows": len(val_ids),
        "fit_episodes": sorted(set(mem_eps[fit_ids].tolist())),
        "validation_episodes": sorted(dev_eps), "seed": seed,
    }
    torch.save(ckpt, out / "magnitude_head.pt")
    with torch.no_grad():
        xx = (x - train["feature_mean"]) / train["feature_std"]
        pred_log = model(torch.as_tensor(xx, dtype=torch.float32)).numpy()
    target_log = np.log(np.clip(y, .25, 4)).astype(np.float32)
    train_json = {
        "target": ckpt["target"], "input_definition": "normalized prefix mean + state + selected unit metadata; no action-history features",
        "query_selection": "frozen Context+history top-1, same-episode excluded",
        "seed": seed, "fit_rows": len(fit_ids), "validation_rows": len(val_ids),
        "fit_episodes": len(set(mem_eps[fit_ids].tolist())), "validation_episodes": len(dev_eps),
        "best_epoch": train["best_epoch"], "validation_logscale_mse": train["val_logscale_mse"],
        "validation_logratio_mae": float(np.mean(np.abs(pred_log[val_ids] - target_log[val_ids]))),
        "validation_target_median": float(np.median(y[val_ids])),
        "validation_prediction_median": float(np.median(np.exp(pred_log[val_ids]))),
        "curve": train["curve"],
    }
    (out / "training.json").write_text(json.dumps(train_json, indent=2) + "\n")

    v2_scale = mh._predict(model, train, xheld)
    old_ck = torch.load(mh.OUT / "magnitude_head.pt", map_location="cpu", weights_only=False)
    old_model = mh._MLP.create(int(old_ck["input_dim"]))
    old_model.load_state_dict(old_ck["state"])
    old_model.eval()
    old_scale = mh._predict(old_model, old_ck, held["x"])
    refs = np.asarray(z["eval_pi05_actions"], np.float32)
    # _metric_rows adds its M1 values under these two names and also emits the
    # common M0/M2/S2/s* controls for each scale list.
    records = mh._metric_rows({"M1_regression": old_scale, "M1_LOTO": v2_scale},
                              held["pick"], pool, z, "heldout", episode_actions, refs=refs)
    metrics = mh._summarize(records)
    np.savez_compressed(out / "offline_rows.npz", heldout_pick=held["pick"],
                        heldout_task=z["heldout_task"], heldout_episode=z["heldout_episode"],
                        heldout_frame=z["heldout_frame"], old_m1_scale=old_scale,
                        magnitude_v2_scale=v2_scale, norm_ratio_target=y,
                        norm_ratio_prediction=mh._predict(model, train, x))
    offline = {"rows": len(held["pick"]), "target": ckpt["target"],
               "methods": metrics,
               "note": "M1_LOTO key stores the new no-history norm-ratio head prediction; no heldout actions entered training. Unit-length metrics use existing magnitude_head._metric_rows; scale is applied once to the six EEF channels."}
    (out / "offline_results.json").write_text(json.dumps(offline, indent=2) + "\n")
    lines = ["# RM-MAG-B offline validation", "",
             f"Memory-fit rows: {len(fit_ids)}; Dev rows: {len(val_ids)}; heldout rows scored: {len(held['pick'])}.",
             f"Target: {ckpt['target']}; best epoch {train['best_epoch']}; Dev log-scale MSE {train['val_logscale_mse']:.6g}.",
             "The selected unit is still chosen by the frozen Context+history key; the new scale head uses prefix+state and selected-unit metadata only.", "",
             "|方法|unit all6 MSE|unit pos MSE|unit rot MSE|final/truth median|scale median|",
             "|---|---:|---:|---:|---:|---:|"]
    for method in ("M1_regression", "M1_LOTO", "S2_pi05_norm", "s_star", "M0_stored", "M2_history_norm"):
        m = metrics.get(method)
        if not m: continue
        lines.append(f"|{method}|{m['unit_all6_mse']:.6f}|{m['unit_pos_mse']:.6f}|{m['unit_rot_mse']:.6f}|{m['ratio_final_over_truth']['median']:.3f}|{m['scale']['median']:.3f}|")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(out), "training": train_json, "offline_rows": len(held["pick"])}, indent=2))


if __name__ == "__main__":
    main()
