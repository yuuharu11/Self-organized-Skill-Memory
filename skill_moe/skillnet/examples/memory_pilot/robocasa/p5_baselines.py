"""Offline P5 history corruption baselines and type-threshold U pools.

This diagnostic deliberately reuses the frozen RM-B P5 checkpoints and
memory pool. Held-out action windows are scoring targets only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import experiment as ex
import expert

STATE = Path(os.environ.get("SKILLNET_STATE_DIR", "/work/.skillnet-state"))
SOURCE = STATE / "outputs/robocasa-memory/memory-01/policy-order-v2"
OUT = STATE / "outputs/robocasa-memory/ctx-01/p5_baselines"


def _write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _history(actions, frame):
    """Return front-padded previous-10 actions and a validity mask."""
    f = int(frame); x = np.zeros((10, 12), np.float32)
    if f < 0:
        return x, np.zeros(10, bool)
    f = min(f, len(actions)); lo = max(0, f - 10)
    if f > lo:
        n = f - lo; x[-n:] = actions[lo:f]
    mask = np.zeros(10, bool)
    if f > lo:
        mask[-(f-lo):] = True
    return x, mask


def _make_queries():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    demos = [d for d in manifest["demos"] if d["split"] == "heldout"]
    episodes = {}
    records = []
    for d in sorted(demos, key=lambda z: (int(z["task_id"]), int(z["episode"]))):
        key = (int(d["task_id"]), int(d["episode"]))
        with np.load(SOURCE / "episodes" / f"task{key[0]}_ep{key[1]:06d}.npz", allow_pickle=False) as z:
            actions = np.asarray(z["action"], np.float32)
            states = np.asarray(z["state"], np.float32)
        episodes[key] = (actions, states)
        for frame in range(0, max(0, len(actions)-9), 10):
            hist, mask = _history(actions, frame)
            target = actions[frame:frame+10]
            if len(target) != 10:
                continue
            shape, _ = ex.encode_shape(target)
            records.append({"task": key[0], "episode": key[1], "global_episode": key[0]*1000+key[1],
                            "frame": frame, "hist": hist, "hist_mask": mask,
                            "state": states[frame], "target": target, "shape": shape})
    if not records:
        raise ValueError("no stride-10 held-out query windows")
    q = {k: np.stack([r[k] for r in records]) for k in ("hist", "hist_mask", "state", "target", "shape")}
    for k in ("task", "episode", "global_episode", "frame"):
        q[k] = np.asarray([r[k] for r in records], np.int32)
    return q, episodes


def _history_conditions(q, episodes):
    n = len(q["task"]); out = {"correct": q["hist"].copy(), "none": np.zeros_like(q["hist"])}
    masks = {"correct": q["hist_mask"].copy(), "none": np.zeros_like(q["hist_mask"])}
    for delta in (-20, -10, 10, 20):
        name = f"shift_{delta:+d}"
        arr = np.zeros_like(q["hist"]); mask = np.zeros_like(q["hist_mask"])
        for i in range(n):
            actions = episodes[(int(q["task"][i]), int(q["episode"][i]))][0]
            arr[i], mask[i] = _history(actions, int(q["frame"][i]) + delta)
        out[name] = arr; masks[name] = mask
    # Use a different held-out episode of the same task, at the same frame.
    out["replace"] = np.zeros_like(q["hist"]); masks["replace"] = np.zeros_like(q["hist_mask"])
    by_task = {}
    for task, ep in episodes:
        by_task.setdefault(task, []).append(ep)
    for i in range(n):
        task, ep = int(q["task"][i]), int(q["episode"][i])
        options = sorted(e for e in by_task[task] if e != ep)
        if not options:
            continue
        other = options[(ep + int(q["frame"][i]) // 10) % len(options)]
        actions = episodes[(task, other)][0]
        out["replace"][i], masks["replace"][i] = _history(actions, int(q["frame"][i]))
    return out, masks


def _p5_predictions(q, histories, masks, pool):
    import torch
    torch.set_num_threads(8)
    from experiment import _features
    from expert import _load_net
    keys = torch.as_tensor(np.asarray(pool["key"], np.float32), dtype=torch.float32)
    results = {cond: [] for cond in histories}
    norms = {}
    # Embed the memory keys once per seed; score all queries from cached
    # embeddings, avoiding repeated candidate-side MLP work.
    for seed in (0, 1, 2):
        net, ck = _load_net(SOURCE / f"models/P5_U_seed{seed}.pt", pool, device="cpu")
        with torch.no_grad():
            ke = torch.nn.functional.normalize(net.key(keys), dim=-1)
            for cond, hist in histories.items():
                query = {"hist": hist, "state": q["state"]}
                x = _features(query, ck["norm"])
                # The frozen checkpoint has no additional mask dimensions.
                # Apply the valid-frame mask directly to the standardized
                # history block, so absent/padded frames contribute zero.
                x[:, :120] = (x[:, :120].reshape(len(x), 10, 12) * masks[cond][..., None]).reshape(len(x), 120)
                pe = []
                for start in range(0, len(x), 256):
                    xt = torch.as_tensor(x[start:start+256], dtype=torch.float32)
                    pe.append(torch.nn.functional.normalize(net.query(xt), dim=-1))
                qe = torch.cat(pe, 0)
                score = (qe @ ke.T * net.log_scale.exp()).numpy()
                pick = score.argmax(axis=1).astype(np.int32)
                results[cond].append(pick)
        norms[str(seed)] = {k: float(v) for k, v in ck["norm"].items() if np.ndim(v) == 0}
    return results, norms


def _unit_gripper_features(pool):
    signs = np.empty(len(pool["length"]), np.int8)
    terminal = np.zeros(len(signs), bool)
    inner = np.zeros(len(signs), np.int16)
    for i in range(len(signs)):
        a, b = int(pool["action_offsets"][i]), int(pool["action_offsets"][i+1])
        grip = np.where(pool["actions_flat"][a:b, 11] >= 0, 1, -1).astype(np.int8)
        if len(grip) == 0:
            signs[i] = 1; continue
        vals, counts = np.unique(grip, return_counts=True)
        signs[i] = int(vals[np.argmax(counts)])  # sorted values make ties choose -1
        inner[i] = int(np.count_nonzero(grip[1:] != grip[:-1]))
        ge = int(pool["global_episode"][i]); next_frame = int(pool["frame"][i]) + (b-a)
        nxt = np.flatnonzero((pool["global_episode"] == ge) & (pool["frame"] == next_frame))
        if len(nxt):
            j = int(nxt[0]); x, y = int(pool["action_offsets"][j]), int(pool["action_offsets"][j+1])
            if y > x:
                terminal[i] = int(np.sign(pool["actions_flat"][b-1, 11])) != int(np.sign(pool["actions_flat"][x, 11]))
    return signs, terminal, inner


def _cluster_threshold(errors, tau):
    """Greedy medoid ball-cover; each assigned member lies within tau of its medoid."""
    e = np.asarray(errors, np.float32)
    d = (e + e.T) * .5
    n = len(d); left = np.ones(n, bool); reps=[]; members=[]
    while left.any():
        ids = np.flatnonzero(left)
        # Largest remaining tau-ball, deterministic smallest-index tie break.
        counts = (d[np.ix_(ids, ids)] <= float(tau)).sum(axis=1)
        rep = int(ids[int(np.argmax(counts))])
        mem = ids[d[rep, ids] <= float(tau)]
        if not len(mem): mem = np.asarray([rep], np.int64)
        reps.append(rep); members.append(mem.astype(np.int32)); left[mem] = False
    return np.asarray(reps, np.int32), members


def build_pools(out=OUT):
    out = Path(out); dest = out / "pools"; dest.mkdir(parents=True, exist_ok=True)
    pool = dict(np.load(SOURCE / "pools/U_memory.npz", allow_pickle=False))
    grip, terminal, inner = _unit_gripper_features(pool)
    np.savez_compressed(dest / "U_gripper_features.npz", unit_id=np.arange(len(grip), dtype=np.int32),
                        grip_sign=grip, terminal_switch=terminal, internal_switch_count=inner)
    errors = np.load(SOURCE / "cache/errors_U_memory.npy", mmap_mode="r")
    result = {"base_units": len(grip), "tau": {}, "gripper": {
        "grip_plus_fraction": float(np.mean(grip > 0)), "terminal_switch_units": int(terminal.sum()),
        "units_with_internal_switch": int((inner > 0).sum()), "internal_switch_count_total": int(inner.sum())}}
    for tau in (.04, .08):
        reps, groups = _cluster_threshold(errors, tau)
        offsets = np.concatenate([[0], np.cumsum([len(x) for x in groups])]).astype(np.int64)
        members = np.concatenate(groups).astype(np.int32)
        outpath = dest / f"U_tau_{tau:.2f}.npz"
        np.savez_compressed(outpath, representative_unit=reps, member_offsets=offsets, member_units=members,
            shape=pool["shape"][reps], key=pool["key"][reps], length=pool["length"][reps],
            task=pool["task"][reps], episode=pool["episode"][reps], frame=pool["frame"][reps],
            grip_sign=grip[reps], terminal_switch=terminal[reps], internal_switch_count=inner[reps])
        result["tau"][str(tau)] = {"cluster_count": int(len(reps)), "median_members": float(np.median([len(x) for x in groups])),
            "p90_members": float(np.quantile([len(x) for x in groups], .9)), "file": outpath.name}
    _write_json(out / "pool_summary.json", result)
    return result


def evaluate(out=OUT):
    import torch
    from experiment import shape_error_matrix
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    q, episodes = _make_queries(); pool = dict(np.load(SOURCE / "pools/U_memory.npz", allow_pickle=False))
    err = shape_error_matrix(q["shape"], pool["shape"])
    histories, masks = _history_conditions(q, episodes)
    picks, _ = _p5_predictions(q, histories, masks, pool)
    task_rates = {}; rows=[]
    for cond, seedpicks in picks.items():
        err_per_seed=[]; task_per_seed=[]; frame0_per_seed=[]
        for seed, pick in enumerate(seedpicks):
            e = err[np.arange(len(q["task"])), pick]
            tm = pool["task"][pick] == q["task"]
            f0 = q["frame"] == 0
            err_per_seed.append(e); task_per_seed.append(tm)
            frame0_per_seed.append((e[f0], tm[f0]))
            rows.append({"condition": cond, "seed": seed, "top1_type_error": e,
                         "task_match": tm, "frame0_error": e[f0], "frame0_task_match": tm[f0],
                         "valid_history_fraction": masks[cond].mean(1)})
        task_rates[cond] = {
            "top1_type_error_mean": float(np.mean(err_per_seed)),
            "seed_mean": [float(np.mean(x)) for x in err_per_seed],
            "task_match_rate": float(np.mean(task_per_seed)),
            "frame0_type_error_mean": float(np.mean([x[0].mean() for x in frame0_per_seed])),
            "frame0_task_match_rate": float(np.mean([x[1].mean() for x in frame0_per_seed])),
            "valid_history_fraction": float(np.mean(masks[cond]))}
    allowed_err = err
    oracle = np.min(allowed_err, axis=1)
    rng = np.random.default_rng(20261111)
    random_ix = rng.integers(0, len(pool["task"]), size=(5, len(q["task"])))
    random_err = np.stack([err[np.arange(len(q["task"])), x] for x in random_ix])
    random_match = np.stack([pool["task"][x] == q["task"] for x in random_ix])
    refs = {"random": {"top1_type_error_mean": float(random_err.mean()), "task_match_rate": float(random_match.mean()),
                        "frame0_type_error_mean": float(random_err[:, q["frame"] == 0].mean()),
                        "frame0_task_match_rate": float(random_match[:, q["frame"] == 0].mean())},
            "oracle": {"top1_type_error_mean": float(oracle.mean()), "task_match_rate": float(pool["task"][err.argmin(1)].__eq__(q["task"]).mean()),
                       "frame0_type_error_mean": float(oracle[q["frame"] == 0].mean()),
                       "frame0_task_match_rate": float((pool["task"][err.argmin(1)] == q["task"])[q["frame"] == 0].mean())}}
    # Save compact row-level, per-seed predictions for downstream context-key comparison.
    pred = {f"pick_{k}": np.stack(v) for k, v in picks.items()}
    np.savez_compressed(out / "p5_baseline_rows.npz", task=q["task"], episode=q["episode"], frame=q["frame"],
        frame0=(q["frame"] == 0), **pred,
        **{f"type_error_{k}": np.stack([r["top1_type_error"] for r in rows if r["condition"] == k]) for k in picks},
        **{f"task_match_{k}": np.stack([r["task_match"] for r in rows if r["condition"] == k]) for k in picks})
    payload = {"n_heldout_episodes": len(episodes), "n_stride10_windows": len(q["task"]),
               "n_frame0": int(np.sum(q["frame"] == 0)), "conditions": task_rates, "references": refs,
               "condition_policy": "frozen RM-B P5 seeds; no held-out tuning; candidate pool is the 80 memory episodes per task",
               "history_mask_note": "For the frozen P5 model (no extra mask dimensions), validity is applied by zeroing masked standardized history timesteps; zero-history therefore has no history contribution. The binary masks are also retained in the row metadata source."}
    _write_json(out / "results.json", payload)
    return payload


def write_report(out=OUT):
    out = Path(out); data = json.loads((out / "results.json").read_text()); pools = json.loads((out / "pool_summary.json").read_text())
    lines = ["実装: 完了", "", "# P5履歴条件ベースライン", "", f"保留120デモから stride 10 で {data['n_stride10_windows']} 窓を作成（frame 0 は {data['n_frame0']} 窓）。正解行動は型誤差の採点とoracle参照だけに使った。P5は既存3 seedを固定して用い、設定選択なし。候補はmemory側80デモのU単位。", "",
             "## P5 履歴条件", "", "| 条件 | 1位型誤差 全体 | frame 0 | タスク一致率 全体 | frame 0 | 履歴有効率 |", "|---|---:|---:|---:|---:|---:|"]
    for k, v in data["conditions"].items():
        lines.append(f"| {k} | {v['top1_type_error_mean']:.6f} | {v['frame0_type_error_mean']:.6f} | {v['task_match_rate']:.3f} | {v['frame0_task_match_rate']:.3f} | {v['valid_history_fraction']:.3f} |")
    for k, v in data["references"].items():
        lines.append(f"| {k} | {v['top1_type_error_mean']:.6f} | {v['frame0_type_error_mean']:.6f} | {v['task_match_rate']:.3f} | {v['frame0_task_match_rate']:.3f} | — |")
    lines += ["", "固定P5の入力次元は既存checkpointと同じに保った。maskは履歴ブロックの標準化後特徴に適用し、無効な時刻をゼロ寄与にした（mask専用の新しい学習済み入力次元は追加していない）。ずれ条件は同一エピソード内の指定frame位置で直前10行動を採った（未来位置へのずれも汚染診断に限り使用）。差し替えは同タスクの別保留エピソード。", "", "## τまとめプール", "", f"元U単位 {pools['base_units']}。グリッパー +1 多数決割合 {pools['gripper']['grip_plus_fraction']:.3f}、終端切替単位 {pools['gripper']['terminal_switch_units']}、内部切替を含む単位 {pools['gripper']['units_with_internal_switch']}（計 {pools['gripper']['internal_switch_count_total']} 回）。型誤差の対称化距離による greedy medoid ball-cover を使用し、各メンバーは代表から τ 以内。", "", "| τ | 代表単位数 | メンバー数中央値 | p90 | 保存ファイル |", "|---:|---:|---:|---:|---|"]
    for k, v in pools["tau"].items(): lines.append(f"| {k} | {v['cluster_count']} | {v['median_members']:.1f} | {v['p90_members']:.1f} | pools/{v['file']} |")
    lines += ["", "型誤差は10フレーム×6軸のS-sharedスケール最適化後誤差。ランダム参照は一様抽出5回平均、oracleは各クエリで最小型誤差の単位。オフライン診断であり、閉ループの性能比較ではない。", ""]
    (out / "report.md").write_text("\n".join(lines))
    return "\n".join(lines)


def run(out=OUT):
    out = Path(out)
    if (out / "results.json").exists():
        raise FileExistsError(f"refusing to overwrite existing results in {out}")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "progress.jsonl").open("a") as f:
        f.write(json.dumps({"stage": "start", "time": time.time()}) + "\n")
    pools = json.loads((out / "pool_summary.json").read_text()) if (out / "pool_summary.json").exists() else build_pools(out)
    result = evaluate(out)
    with (out / "progress.jsonl").open("a") as f:
        f.write(json.dumps({"stage": "complete", "time": time.time(), "rows": result["n_stride10_windows"]}) + "\n")
    return write_report(out)


if __name__ == "__main__":
    print(run())
