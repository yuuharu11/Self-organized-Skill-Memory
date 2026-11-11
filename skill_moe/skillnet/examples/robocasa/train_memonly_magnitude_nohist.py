"""Fit a history-free magnitude head on the existing memory-side split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import torch

    from magnitude_head import (
        CTX, KEYLITE, SOURCE, _MLP, _fit, _load_keylite, _make_rows,
        _unit_metadata,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-epochs", type=int, default=150)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(20261111)

    rk = _load_keylite()
    z = np.load(CTX / "cache/rows.npz", allow_pickle=False)
    pool = dict(np.load(CTX / "cache/pool.npz", allow_pickle=False))
    feat = np.asarray(np.load(CTX / "cache/prefix_mean_fp16.npy", mmap_mode="r"), np.float32)
    ck = torch.load(KEYLITE / "context_history.pt", map_location="cpu", weights_only=False)
    if not ck.get("input_history", False):
        raise AssertionError("retrieval source must remain the Context+history model")

    # Keep exactly the original magnitude task's top-1 retrieval and target
    # labels. Only the magnitude head's input loses action history.
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    import experiment as ex
    episode_actions = {
        (int(d["task_id"]), int(d["episode"])):
            ex._load_cached_episode(SOURCE, int(d["task_id"]), int(d["episode"]))[0]
        for d in manifest["demos"]
    }
    mem = _make_rows(rk, ck, z, pool, feat, "memory", episode_actions)
    q_context, _ = rk.assemble_context(z, feat, ck["norm"], "memory")
    uk, _ = _unit_metadata(pool)
    x = np.concatenate((np.asarray(q_context, np.float32), uk[mem["pick"]]), axis=1).astype(np.float32)
    y = np.asarray(mem["s_star"], np.float32)
    episodes = np.asarray(z["memory_episode"], np.int64)
    dev = set(map(int, ck["dev_episodes"]))
    val_mask = np.asarray([int(e) in dev for e in episodes])
    fit_ids, val_ids = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
    if not len(fit_ids) or not len(val_ids):
        raise AssertionError("empty train/dev partition")

    model, trained = _fit(x, y, fit_ids, val_ids, seed=20261111,
                          max_epochs=args.max_epochs, label="history_free")
    ckpt = {
        "state": {k: v.detach().cpu() for k, v in trained["state"].items()},
        "feature_mean": trained["feature_mean"], "feature_std": trained["feature_std"],
        "best_epoch": trained["best_epoch"], "val_logscale_mse": trained["val_logscale_mse"],
        "input_dim": int(x.shape[1]), "target": "log clipped position s*",
        "history_free": True, "fit_rows": len(fit_ids), "validation_rows": len(val_ids),
        "fit_episodes": sorted(set(episodes[fit_ids].tolist())),
        "validation_episodes": sorted(dev),
    }
    torch.save(ckpt, out / "magnitude_head_nohist.pt")
    curve = [{k: float(v) for k, v in row.items()} for row in trained["curve"]]
    (out / "training.json").write_text(json.dumps({
        "input_dim": x.shape[1], "query_context_dim": q_context.shape[1],
        "unit_metadata_dim": uk.shape[1], "fit_rows": len(fit_ids),
        "validation_rows": len(val_ids), "fit_episodes": len(set(episodes[fit_ids].tolist())),
        "validation_episodes": len(dev), "best_epoch": trained["best_epoch"],
        "validation_logscale_mse": trained["val_logscale_mse"], "curve": curve,
    }, indent=2) + "\n")
    print(f"rows={len(x)} fit={len(fit_ids)} dev={len(val_ids)} best_epoch={trained['best_epoch']} val={trained['val_logscale_mse']:.6g}")


if __name__ == "__main__":
    main()
