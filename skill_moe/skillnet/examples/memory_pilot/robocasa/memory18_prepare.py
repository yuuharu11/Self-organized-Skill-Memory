"""Prepare the 18-task RoboCasa atomic memory and resumable context cache.

This is preparation only: no ranking, magnitude, gripper, or bridge training.
It reuses the reviewed memory-01 unit and context implementations.
"""
from __future__ import annotations

import argparse
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

STATE = Path(os.environ.get("SKILLNET_STATE_DIR", "/work/.skillnet-state"))
OUT = STATE / "outputs/robocasa-memory/memory-18"
SOURCE = STATE / "outputs/robocasa-memory/memory-01"
TASK_SET_FILE = Path("/work/data_process/robocasa/task_sets.json")
CHECKPOINT = STATE / "checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000"


def _append_progress(path: Path, **record):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
    print(json.dumps(row, ensure_ascii=False), flush=True)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=ex._json_default) + "\n")
    tmp.replace(path)


def canonical_tasks():
    registry = json.loads(TASK_SET_FILE.read_text())
    tasks = tuple(registry["task_sets"]["atomic_seen"])
    if len(tasks) != 18 or len(set(tasks)) != 18:
        raise AssertionError(f"atomic_seen task registry should contain 18 unique tasks, got {len(tasks)}")
    return tasks


def _existing_splits():
    manifest_path = SOURCE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return {str(t["task"]): {
        "memory": np.asarray(t["memory_episodes"], np.int64),
        "heldout": np.asarray(t["heldout_episodes"], np.int64),
        "old_task_id": int(t["task_id"]),
    } for t in manifest["task_splits"]}, manifest


def prepare_memory(out=OUT):
    """Use the original six exact split assignments, and deterministic 80/20 elsewhere."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = canonical_tasks()
    old_splits, old_manifest = _existing_splits()
    if (out / "manifest.json").exists():
        manifest = json.loads((out / "manifest.json").read_text())
        if tuple(manifest["tasks"]) != tasks or int(manifest["demo_limit"]) != ex.N_DEMOS:
            raise ValueError("existing memory-18 manifest disagrees with canonical task order/configuration")
        ex.build_pools(out)
        return manifest

    original_splitter = ex.split_episode_ids

    def compatible_splitter(ids, *, seed=ex.SPLIT_SEED):
        ix = int(seed) - int(ex.SPLIT_SEED)
        if ix < 0 or ix >= len(tasks):
            raise ValueError(f"unexpected task split seed {seed}")
        task = tasks[ix]
        if task in old_splits:
            row = old_splits[task]
            # The exact memory/held-out IDs from memory-01 are retained, not redrawn.
            return row["memory"].copy(), row["heldout"].copy()
        return original_splitter(ids, seed=int(ex.SPLIT_SEED) + ix)

    ex.split_episode_ids = compatible_splitter
    try:
        manifest = ex.prepare(out, tasks=tasks, demo_limit=ex.N_DEMOS)
    finally:
        ex.split_episode_ids = original_splitter

    manifest["task_set"] = "atomic_seen"
    manifest["canonical_task_order_source"] = str(TASK_SET_FILE)
    manifest["split_rule"] = (
        "First 100 numeric episode IDs per task; 80 memory/train and 20 held-out. "
        "For the six memory-01 tasks, preserve the exact original memory-01 episode assignment. "
        "For the other 12 tasks, use the original per-task seeded 80/20 permutation "
        "with seed=20261110+canonical atomic_seen task index."
    )
    manifest["six_task_split_source"] = str(SOURCE / "manifest.json")
    _write_json(out / "manifest.json", manifest)
    ex.build_pools(out)
    _append_progress(out / "progress.jsonl", stage="memory_complete", tasks=len(tasks),
                     episodes=len(manifest["demos"]))
    return manifest


def _compare_six_task_pools(out=OUT):
    """Verify exact episode, boundary, action and DCT agreement for the six prior tasks."""
    old_manifest = json.loads((SOURCE / "manifest.json").read_text())
    new_manifest = json.loads((Path(out) / "manifest.json").read_text())
    old_id = {str(x["task"]): int(x["task_id"]) for x in old_manifest["task_splits"]}
    new_id = {str(x["task"]): int(x["task_id"]) for x in new_manifest["task_splits"]}
    result = {}
    for family in ("U", "W"):
        old = dict(np.load(SOURCE / f"pools/{family}_memory.npz", allow_pickle=False))
        new = dict(np.load(Path(out) / f"pools/{family}_memory.npz", allow_pickle=False))
        family_rows = {}
        for task in old_manifest["tasks"]:
            oi = np.flatnonzero(old["task"] == old_id[task])
            ni = np.flatnonzero(new["task"] == new_id[task])
            old_by = {(int(old["episode"][i]), int(old["frame"][i])): int(i) for i in oi}
            new_by = {(int(new["episode"][i]), int(new["frame"][i])): int(i) for i in ni}
            same_keys = old_by.keys() == new_by.keys()
            equal = bool(same_keys)
            if equal:
                for key in old_by:
                    a, b = old_by[key], new_by[key]
                    if int(old["length"][a]) != int(new["length"][b]) or not np.array_equal(old["shape"][a], new["shape"][b]):
                        equal = False; break
                    oa, ob = int(old["action_offsets"][a]), int(old["action_offsets"][a+1])
                    na, nb = int(new["action_offsets"][b]), int(new["action_offsets"][b+1])
                    if not np.array_equal(old["actions_flat"][oa:ob], new["actions_flat"][na:nb]):
                        equal = False; break
            old_split = next(x for x in old_manifest["task_splits"] if x["task"] == task)
            new_split = next(x for x in new_manifest["task_splits"] if x["task"] == task)
            split_equal = (old_split["memory_episodes"] == new_split["memory_episodes"] and
                           old_split["heldout_episodes"] == new_split["heldout_episodes"])
            family_rows[task] = {"old_rows": int(len(oi)), "new_rows": int(len(ni)),
                                 "same_episode_frame_keys": bool(same_keys),
                                 "same_split": bool(split_equal), "identical_units": bool(equal and split_equal)}
        result[family] = family_rows
    if not all(v["identical_units"] for family in result.values() for v in family.values()):
        bad = [(f, t, d) for f, rows in result.items() for t, d in rows.items() if not d["identical_units"]]
        raise AssertionError(f"memory-01 six-task subset mismatch: {bad[:4]}")
    return result


def _context_rows(out=OUT):
    # Import after setting SOURCE so the shared implementation inventories memory-18.
    import context_key as ck
    ck.SOURCE = Path(out)
    ck.CHECKPOINT = CHECKPOINT
    manifest, pool, rows, candidates = ck._inventory()
    contexts = {}; references = {}
    for split in ("memory", "heldout"):
        for i, row in enumerate(rows[split]):
            key = ck._ctx_id(row); contexts.setdefault(key, row); references[f"{split}:{i}"] = key
    for i, row in enumerate(candidates):
        key = ck._ctx_id(row); contexts.setdefault(key, row); references[f"candidate:{i}"] = key
    keys = list(contexts)
    lookup = {key: i for i, key in enumerate(keys)}
    context_ids = np.asarray([[int(contexts[k]["task"]), int(contexts[k]["episode"]), int(contexts[k]["frame"])] for k in keys], np.int32)
    return ck, manifest, pool, rows, candidates, contexts, references, lookup, context_ids


def extract_context(out=OUT, *, batch_size=4):
    """Extract context features to resumable fp16 memmaps, no model training.

    This preparation caches prefix features only. It intentionally does not
    run an extra π0.5 action sampler over held-out windows.
    """
    import jax
    from openpi import transforms
    import context_key as ck
    out = Path(out); ctx_out = out / "ctx"; cache = ctx_out / "cache"
    ctx_out.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    ck, manifest, pool, rows, candidates, contexts, refs, key_to_ix, context_ids = _context_rows(out)
    index_path = cache / "feature_index.npz"
    if index_path.exists():
        old = dict(np.load(index_path, allow_pickle=False))
        if not np.array_equal(old["context_ids"], context_ids):
            raise ValueError("context index differs from resumable feature cache; refusing to overwrite")
    else:
        np.savez_compressed(index_path, context_ids=context_ids,
            memory_context=np.asarray([key_to_ix[refs[f"memory:{i}"]] for i in range(len(rows["memory"]))], np.int32),
            heldout_context=np.asarray([key_to_ix[refs[f"heldout:{i}"]] for i in range(len(rows["heldout"]))], np.int32),
            candidate_context=np.asarray([key_to_ix[refs[f"candidate:{i}"]] for i in range(len(candidates))], np.int32))
    nctx = len(context_ids); nheld = len(rows["heldout"])
    paths = {"final": cache / "prefix_mean_fp16.npy", "input": cache / "prefix_input_mean_fp16.npy",
             "done": cache / "feature_done.npy"}
    def open_array(path, shape, dtype):
        if path.exists():
            arr = np.load(path, mmap_mode="r+")
            if arr.shape != shape or arr.dtype != np.dtype(dtype):
                raise ValueError(f"incompatible partial cache {path}: {arr.shape}/{arr.dtype}; refusing overwrite")
            return arr
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    final = open_array(paths["final"], (nctx, 2048), np.float16)
    inp = open_array(paths["input"], (nctx, 2048), np.float16)
    done = open_array(paths["done"], (nctx,), np.bool_)
    # Avoid loading a large policy checkpoint when resuming only to finalize
    # artifacts after every feature row has already been persisted.
    need_extraction = not bool(np.all(done))
    if need_extraction:
        transform = ck._make_transforms(CHECKPOINT)
        model, _cfg = ck.expert._load_expert_model(CHECKPOINT)
        extractor = ck._prefix_extractor(model)
    else:
        transform = model = extractor = None
    held_rows = rows["heldout"]
    started = time.monotonic(); progress = ctx_out / "progress.jsonl"
    # Held-out observations are processed first to persist the matching regular-policy reference.
    for start in range(0, nheld, int(batch_size)):
        stop = min(start + int(batch_size), nheld)
        ids = np.asarray([key_to_ix[ck._ctx_id(r)] for r in held_rows[start:stop]], np.int64)
        need_feat = ~np.asarray(done[ids])
        if not need_feat.any():
            continue
        feat, pre, actions = ck._feature_rows(held_rows[start:stop], transform, extractor,
            batch_size=batch_size, predict_actions=False, model=model)
        for j, ix in enumerate(ids):
            if not done[ix]: final[ix] = feat[j]; inp[ix] = pre[j]; done[ix] = True
        if start % 256 == 0:
            final.flush(); inp.flush(); done.flush()
            _append_progress(progress, stage="ctx_extract", split="heldout", rows_done=int(done.sum()),
                             contexts=nctx, elapsed_sec=round(time.monotonic()-started, 1))
    # Other memory-window and candidate contexts; deduplicated by task/episode/frame.
    held_keys = {ck._ctx_id(r) for r in held_rows}
    remaining = [(k, v) for k, v in contexts.items() if k not in held_keys]
    pending = [(k, v) for k, v in remaining if not done[key_to_ix[k]]]
    for start in range(0, len(pending), int(batch_size)):
        chunk = pending[start:start+int(batch_size)]
        feat, pre, _ = ck._feature_rows([v for _, v in chunk], transform, extractor,
            batch_size=batch_size, model=model)
        for j, (key, _row) in enumerate(chunk):
            ix = key_to_ix[key]; final[ix] = feat[j]; inp[ix] = pre[j]; done[ix] = True
        final.flush(); inp.flush(); done.flush()
        if start % 256 == 0 or start + len(chunk) == len(pending):
            _append_progress(progress, stage="ctx_extract", split="memory_and_candidates",
                             rows_done=int(done.sum()), contexts=nctx,
                             elapsed_sec=round(time.monotonic()-started, 1))
    final.flush(); inp.flush(); done.flush()
    if not np.all(done):
        raise RuntimeError(f"context extraction incomplete: features={int(done.sum())}/{nctx}")

    # Write the context-key compatible row table and exact memory pool.
    tables = {"candidate_context": np.asarray([key_to_ix[refs[f"candidate:{i}"]] for i in range(len(candidates))], np.int32)}
    for split in ("memory", "heldout"):
        rec = rows[split]
        tables[f"{split}_context"] = np.asarray([key_to_ix[refs[f"{split}:{i}"]] for i in range(len(rec))], np.int32)
        tables[f"{split}_task"] = np.asarray([r["task"] for r in rec], np.int16)
        tables[f"{split}_episode"] = np.asarray([r["global_episode"] for r in rec], np.int32)
        tables[f"{split}_local_episode"] = np.asarray([r["episode"] for r in rec], np.int32)
        tables[f"{split}_frame"] = np.asarray([r["frame"] for r in rec], np.int32)
        tables[f"{split}_state"] = np.stack([ex.state_storage_to_policy(r["state"]) for r in rec]).astype(np.float32)
        tables[f"{split}_storage_state"] = np.stack([r["state"] for r in rec]).astype(np.float32)
        tables[f"{split}_hist"] = np.stack([r["hist"] for r in rec]).astype(np.float32)
        tables[f"{split}_target"] = np.stack([r["target"] for r in rec]).astype(np.float32)
    for field in ("task", "episode", "global_episode", "frame"):
        tables[f"candidate_{field}"] = np.asarray([r["task" if field == "task" else field] for r in candidates], np.int32)
    candidate_state = []
    episode_cache = {}
    for r in candidates:
        ek = (int(r["task"]), int(r["episode"]))
        if ek not in episode_cache: episode_cache[ek] = ex._load_cached_episode(out, *ek)[1]
        candidate_state.append(episode_cache[ek][int(r["frame"])])
    tables["candidate_state"] = ex.state_storage_to_policy(np.stack(candidate_state)).astype(np.float32)
    tables["candidate_storage_state"] = np.stack(candidate_state).astype(np.float32)
    # Episode numbers are local to a task; comparing the episode column alone
    # creates false positives across tasks.  Compare the stable task/global-id
    # pair instead.
    candidate_episode_keys = set(zip(tables["candidate_task"].tolist(),
                                     tables["candidate_global_episode"].tolist()))
    heldout_episode_keys = set(zip(tables["heldout_task"].tolist(),
                                   tables["heldout_episode"].tolist()))
    if candidate_episode_keys & heldout_episode_keys:
        raise AssertionError("held-out episodes leaked into memory candidate contexts")
    tables["heldout_grip_sign"] = np.asarray([1 if r["target"][0, 11] >= 0 else -1 for r in held_rows], np.int8)
    tables["memory_grip_sign"] = np.asarray([1 if r["target"][0, 11] >= 0 else -1 for r in rows["memory"]], np.int8)
    rows_file = cache / "rows.npz"
    pool_file = cache / "pool.npz"
    for path, arrays in ((rows_file, tables), (pool_file, pool)):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing completed cache {path}")
        np.savez_compressed(path, **arrays)
    _write_json(cache / "feature_manifest.json", {
        "checkpoint": str(CHECKPOINT), "source": str(out),
        "feature": "mean of valid final PaliGemma prefix hidden tokens and pre-Transformer prefix embeddings; left/right/wrist images plus task prompt",
        "storage": "fp16 per-context feature memmaps; cast to float32 by downstream models",
        "contexts": int(nctx), "memory_stride10_windows": int(len(rows["memory"])),
        "heldout_stride10_windows": int(len(rows["heldout"])), "candidates": int(len(candidates)),
        "batch_size": int(batch_size), "heldout_pi05_action_sampler": "not run; prefix features only",
        "resumable": True, "feature_files": [paths["final"].name, paths["input"].name]})
    _write_json(cache / "feature_complete.json", {"contexts": int(nctx), "features_complete": True,
        "heldout_pi05_actions_complete": False, "rows_file": str(rows_file), "pool_file": str(pool_file)})
    _append_progress(progress, stage="ctx_complete", contexts=nctx,
                     memory_windows=len(rows["memory"]), heldout_windows=len(rows["heldout"]),
                     candidates=len(candidates), elapsed_sec=round(time.monotonic()-started, 1))
    return {"contexts": nctx, "memory_windows": len(rows["memory"]),
            "heldout_windows": len(rows["heldout"]), "candidates": len(candidates)}


def summary(out=OUT):
    out = Path(out); manifest = json.loads((out / "manifest.json").read_text())
    complete = json.loads((out / "pools/complete.json").read_text())
    equality = _compare_six_task_pools(out)
    rows = []
    for family in ("U", "W"):
        pool = np.load(out / f"pools/{family}_memory.npz", allow_pickle=False)
        for tid, task in enumerate(manifest["tasks"]):
            ids = np.flatnonzero(pool["task"] == tid); lengths = np.asarray(pool["length"][ids], np.float64)
            rows.append({"family": family, "task": task, "items": int(len(ids)),
                "episodes": int(len(np.unique(pool["episode"][ids]))),
                "length_frames_p10_median_p90": [float(np.quantile(lengths,.1)), float(np.median(lengths)), float(np.quantile(lengths,.9))] if len(ids) else [],
                "length_seconds_p10_median_p90": [float(np.quantile(lengths,.1)/20), float(np.median(lengths)/20), float(np.quantile(lengths,.9)/20)] if len(ids) else []})
    ctx_manifest = out / "ctx/cache/feature_manifest.json"
    feature = json.loads(ctx_manifest.read_text()) if ctx_manifest.exists() else None
    result = {"task_order": manifest["tasks"], "episodes": len(manifest["demos"]),
        "memory_per_task": manifest["memory_per_task"], "heldout_per_task": manifest["heldout_per_task"],
        "memory_pool": complete, "task_length_rows": rows, "six_task_exact_unit_match": equality,
        "ctx_features": feature, "training_status": "not run by design"}
    _write_json(out / "summary.json", result)
    status = "実装: 完了" if feature is not None and (out/"ctx/cache/feature_complete.json").exists() else "実装: 未完了（文脈特徴抽出が未完了）"
    lines = [status, "", "# RoboCasa memory-18 準備", "",
        "18タスクのatomic_seenデータを、canonical task-set順で処理した。6タスク分はmemory-01の同じ80/20エピソード割当を再利用し、新規12タスクはcanonical task index由来の固定seedで80/20分割。メモリ構築・VLM特徴準備のみで、キー・大きさ・グリッパー・Expertの学習は実行していない。", "",
        f"データ: {len(manifest['demos'])} demos（各タスク100、memory 80 / heldout 20）。action 12次元のLeRobot順、state 16次元、20Hz。base非ゼロ率(memory)={complete['base_nonzero_fraction_memory']:.6f}、mode非ゼロ率(memory)={complete['mode_nonzero_fraction_memory']:.6f}。", "",
        "## U / W のタスク別プール", "", "|プール|タスク|件数|エピソード数|長さ p10 / median / p90 (frame)|長さ p10 / median / p90 (秒)|", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        a, b, c = r["length_frames_p10_median_p90"]; d, e, f = r["length_seconds_p10_median_p90"]
        lines.append(f"|{r['family']}|{r['task']}|{r['items']}|{r['episodes']}|{a:.1f} / {b:.1f} / {c:.1f}|{d:.2f} / {e:.2f} / {f:.2f}|")
    lines += ["", "## memory-01 との一致", "", "|プール|タスク|旧/新件数|分割一致|各単位の開始・長さ・生指令・DCT形状一致|", "|---|---|---:|---|---|"]
    for family, tasks in equality.items():
        for task, v in tasks.items():
            lines.append(f"|{family}|{task}|{v['old_rows']}/{v['new_rows']}|{v['same_split']}|{v['identical_units']}|")
    lines += ["", "## 文脈特徴", ""]
    if feature:
        lines.append(f"特徴抽出完了: {feature['contexts']} unique contexts（memory windows {feature['memory_stride10_windows']}、heldout windows {feature['heldout_stride10_windows']}、候補単位開始点 {feature['candidates']}）。最終層 prefix と Transformer 入力 prefix の各2048次元平均を fp16 で保存した。保留窓の正解行動は採点用の行テーブルに分離してある。追加のπ0.5行動サンプラーは準備範囲外なので回していない。特徴は小分けのmemmap書込みとdone maskで再開可能。")
    else:
        lines.append("文脈特徴抽出は未完了。`ctx/progress.jsonl` と `ctx/cache/feature_done.npy` を見て同じコマンドで再開できる。")
    lines += ["", "## 次工程の時間見積もり", "",
        "暫定の工数/実時間見積もり（今回の18タスク出力を基準に再調整可能）: retrieval key 3 seed 約1–2時間、magnitude head 約20–40分、gripper head 約20–40分、Expert下書き作成 約30–60分、Expert 3,000更新 約1–2時間、18×50閉ループは1サーバー・16 workersで約3–6時間（GPU/シミュレータ競合により変動）。学習・閉ループは今回回していない。", "",
        "制約: 画像prefixの特徴作成には pi05_robocasa_multitask の75000 checkpointを使用。保留デモの行動は採点用テーブルに保持し、memory候補・学習対象には含めていない。今回はモデル学習も閉ループ評価も実施していない。", "",
        "実行の再開: `source /work/env/activate.sh && export SKILLNET_STATE_DIR=/work/.skillnet-state XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.18 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=/work/.skillnet-state/worktrees/robocasa-memory/skill_moe/skillnet/examples/memory_pilot/robocasa:/work/.skillnet-state/worktrees/robocasa-memory/skill_moe/skillnet/examples/robocasa && taskset -c 8-15 nice -n 10 /work/.skillnet-state/envs/skillnet/bin/python /work/.skillnet-state/worktrees/robocasa-memory/skill_moe/skillnet/examples/memory_pilot/robocasa/memory18_prepare.py all`", ""]
    (out / "report.md").write_text("\n".join(lines))
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=("memory", "ctx", "all", "report")); p.add_argument("--batch-size", type=int, default=4)
    a = p.parse_args(); out = OUT
    if a.stage in ("memory", "all"):
        prepare_memory(out)
    if a.stage in ("ctx", "all"):
        if not (out / "pools/complete.json").exists():
            raise FileNotFoundError("memory pool is incomplete; run memory stage first")
        extract_context(out, batch_size=a.batch_size)
    summary(out)


if __name__ == "__main__":
    main()
