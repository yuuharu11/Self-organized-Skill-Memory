#!/usr/bin/env python3
"""Summarize RM-SPEED without modifying prior experiment artifacts."""
import json
from pathlib import Path
import sys
import numpy as np

out = Path(sys.argv[1])
conds = ("x1p5", "x2p0", "nohist")
tasks = ("PickPlaceCounterToCabinet", "OpenDrawer", "CloseFridge")
root = Path("/work/.skillnet-state/outputs/robocasa-memory")
base = json.loads((root / "baseline-01/v6-nosubtask/eval/results.json").read_text())
base_map = {(r["task"], int(r["trial"])): r for r in base["records"]}
prior_records = []
for prior_file in ("server-a", "server-b"):
    prior_records.extend(json.loads((root / f"closed-01/memonly-01/eval/{prior_file}/results.json").read_text())["records"])
prior_map = {(r["task"], int(r["trial"])): r for r in prior_records}

def med(values):
    return float(np.median(values)) if values else float("nan")

lines = ["実装: 未完了", "", "# RM-SPEED: 大きさ倍率の切り分け", "",
         "|条件|タスク|成功|memonly-01 同trial|baseline 同trial|成功steps中央値|倍率中央値|実行位置ノルム/step中央値|", "|---|---|---:|---:|---:|---:|---:|---:|"]
all_complete = True
for cond in conds:
    result_path = out / cond / "eval/results.json"
    if not result_path.exists():
        all_complete = False
        continue
    data = json.loads(result_path.read_text())
    recs = {(r["task"], int(r["trial"])): r for r in data["records"]}
    if len(recs) != 36:
        all_complete = False
    logs = []
    qpath = out / cond / "queries.jsonl"
    if qpath.exists():
        for line in qpath.read_text().splitlines():
            try: item = json.loads(line)
            except json.JSONDecodeError: continue
            if item.get("record_type") == "memory_only_query": logs.append(item)
    summaries = {}
    for task in tasks:
        qrows = [q for q in logs if str(q.get("episode_id", "")).startswith(task + ":")]
        qscales = [float(u["magnitude"]) for q in qrows for u in q.get("selected_units", [])]
        qnorms = [float(q["position_step_norm_median"]) for q in qrows if "position_step_norm_median" in q]
        summaries[task] = (med(qscales), med(qnorms))
    for task in tasks:
        rows = [r for (t, _), r in recs.items() if t == task]
        succ = sum(bool(r["success"]) for r in rows)
        b = [base_map[(task, trial)] for (t, trial) in recs if t == task and (task, trial) in base_map]
        old = [prior_map[(task, trial)] for (t, trial) in recs if t == task and (task, trial) in prior_map]
        success_steps = [int(r["steps"]) for r in rows if r["success"]]
        scale_med, norm_med = summaries[task]
        lines.append(f"|{cond}|{task}|{succ}/{len(rows)}|{sum(bool(r['success']) for r in old)}/{len(old)}|{sum(bool(r['success']) for r in b)}/{len(b)}|{med(success_steps):.1f}|{scale_med:.3f}|{norm_med:.5f}|")

status = (out / "eval-status.txt").read_text() if (out / "eval-status.txt").exists() else ""
all_complete = all_complete and "evaluation_complete=" in status
lines[0] = "実装: 完了" if all_complete else "実装: 未完了"
train_file = out / "nohist-training/training.json"
if train_file.exists():
    tr = json.loads(train_file.read_text())
    lines += ["", f"NH magnitude head: fit={tr['fit_rows']} rows / {tr['fit_episodes']} episodes; Dev={tr['validation_rows']} rows / {tr['validation_episodes']} episodes; best epoch={tr['best_epoch']}, Dev log-scale MSE={tr['validation_logscale_mse']:.6g}."]
lines += ["", "倍率は予測値に条件倍率を掛けた後 [.25,4] に clip。実行位置ノルムは各問い合わせで返したチャンクの先頭 replan=5 行における、正規化 fork action の位置3次元 L2 norm（問い合わせ中央値の集計）。ベースラインの action-query trace は保存されておらず、π0.5 の同じ seed での位置ノルムは取得不能。成功数と steps はベースラインの同trial（seed規則一致）と比較。", "", "全36試行/条件が完了していない場合、途中経過・worker shard を残しており、この表は完了した条件だけを含む。"]
(out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"complete={all_complete}")
