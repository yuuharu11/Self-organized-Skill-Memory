"""Result-first summaries. Incomplete episodes are never policy failures."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


CONDITIONS = ["original", "time", "achievement", "effect", "oracle"]


def paired_statistics(rows, first, second):
    differences = defaultdict(list)
    rescue = harm = 0
    pairs = []
    for key, arms in sorted(rows.items()):
        if first not in arms or second not in arms:
            continue
        a, b = arms[first]["success"], arms[second]["success"]
        delta = int(a)-int(b)
        differences[key[0]].append(delta)
        rescue += delta == 1
        harm += delta == -1
        pairs.append(dict(task=key[0]+1, init=key[1], first=a, second=b, difference=delta))
    if not pairs:
        return dict(n=0, first=first, second=second)
    rng = np.random.default_rng(20261103)
    tasks = sorted(differences)
    draws = []
    for _ in range(4000):
        selected = rng.choice(tasks, len(tasks), replace=True)
        draws.append(np.mean([np.mean(rng.choice(differences[int(task)], len(differences[int(task)]), replace=True)) for task in selected]))
    return dict(first=first, second=second, n=len(pairs), rescue=rescue, harm=harm,
        difference=float(np.mean([row["difference"] for row in pairs])),
        hierarchical_bootstrap_95=np.quantile(draws, [.025,.975]).tolist(),
        task_differences={str(task+1):float(np.mean(values)) for task, values in differences.items()}, pairs=pairs)


def summarize(root):
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    matched = defaultdict(dict)
    outcomes = []
    for result in sorted((root / "evaluation").glob("*/result.json")):
        row = json.loads(result.read_text())
        job = row["job"]
        matched[job["task"],job["init"]][job["condition"]] = row
        queries = [json.loads(line) for line in (result.parent/"queries.jsonl").read_text().splitlines()]
        switch = next((q for q in queries if q["switched"]), None)
        first_verified = next((q["step"] for q in queries if q["truth"]["verified"][0]), None)
        monitor_queries = [q for q in queries if q["step"]%10==0]
        outcomes.append(dict(task=job["task"]+1, init=job["init"], condition=job["condition"],
            success=row["success"], end_step=row["end_step"], switches=row["switches"], rollbacks=row["rollbacks"],
            switch_step=None if switch is None else switch["step"],
            switch_while_unverified=None if switch is None else not switch["truth"]["verified"][0],
            delay_from_first_sampled_verified=None if switch is None or first_verified is None else switch["step"]-first_verified,
            monitor_s=sum(q["decision"].get("monitor_s",0) for q in monitor_queries),
            inference_s=sum(q["inference_s"] for q in queries), wall_s=row["wall_s"]))
    complete_five = {key:arms for key,arms in matched.items() if set(arms)==set(CONDITIONS)}
    summary = dict(updated_utc=datetime.now(timezone.utc).isoformat(), planned_episodes=90,
        completed_episodes=len(outcomes), complete_five_condition_initial_states=len(complete_five),
        conditions={}, common_five_conditions={}, comparisons={}, held_controller_tasks={}, outcomes=outcomes)
    for condition in CONDITIONS:
        own=[row for row in outcomes if row["condition"]==condition]
        common=[arms[condition] for arms in complete_five.values()]
        summary["conditions"][condition]=dict(n=len(own), successes=sum(row["success"] for row in own),
            switches=sum(row["switches"] for row in own), rollbacks=sum(row["rollbacks"] for row in own))
        summary["common_five_conditions"][condition]=dict(n=len(common), successes=sum(row["success"] for row in common))
    for first,second in [(c,"original") for c in CONDITIONS if c!="original"]+[("effect","achievement")]:
        summary["comparisons"][first+"_vs_"+second]=paired_statistics(matched,first,second)
    held={key:arms for key,arms in matched.items() if key[0] in (5,8)}
    summary["held_controller_tasks"]=paired_statistics(held,"effect","achievement")
    summary["parents_complete"]=len(list((root/"parents").glob("*/result.json")))
    summary["paired_arms_complete"]=len(list((root/"pairs").glob("*/result.json")))
    summary["incomplete_attempts"]=[]
    for job in root.glob("*/*/job.json"):
        if not (job.parent/"result.json").exists():
            summary["incomplete_attempts"].append(str(job.parent.relative_to(root)))
    report_path=analysis/"summary.json"
    tmp=report_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    tmp.replace(report_path)
    if outcomes:
        with (analysis/"episodes.csv").open("w") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(outcomes[0]));writer.writeheader();writer.writerows(outcomes)
    lines=["# 指示更新の半日予備実験：実測", "", f"更新UTC: {summary['updated_utc']}", "",
        "提案：凍結VLAに対し、達成確認と予測した介入効果に基づいて残り指示へ更新する。", "",
        f"完了 {len(outcomes)}/90。5条件が揃う初期状態 {len(complete_five)}/18。未完了・実行エラーを方策失敗に数えない。",
        "", "## 5条件が共通に完了した初期状態だけの比較", "", "| 条件 | 成功 / 試行 | 成功率 |", "|---|---:|---:|"]
    for condition in CONDITIONS:
        value=summary["common_five_conditions"][condition]
        rate=f"{100*value['successes']/value['n']:.1f}%" if value["n"] else "未集計"
        lines.append(f"| {condition} | {value['successes']} / {value['n']} | {rate} |")
    lines += ["", "## 対応あり比較", "", "各比較で両条件が完了した同じ初期状態だけを使用。区間はタスク→初期状態の階層bootstrapで、学習seed間のばらつきは含まない。", "",
              "| 比較 | 組数 | 救済 | 悪化 | 成功率差 | 95%区間 |", "|---|---:|---:|---:|---:|---|"]
    for name,row in summary["comparisons"].items():
        if row["n"]:
            lo,hi=row["hierarchical_bootstrap_95"]
            lines.append(f"| {name} | {row['n']} | {row['rescue']} | {row['harm']} | {100*row['difference']:+.1f}pt | [{100*lo:+.1f}, {100*hi:+.1f}] |")
    lines += ["", "## タスク別：共通5条件", "", "| タスク | 初期状態数 | original | time | achievement | effect | oracle |", "|---|---:|---:|---:|---:|---:|---:|"]
    for task in range(9):
        arms=[value for key,value in complete_five.items() if key[0]==task]
        lines.append("| "+" | ".join([f"{task+1:02d}",str(len(arms))]+[str(sum(a[c]["success"] for a in arms)) for c in CONDITIONS])+" |")
    lines += ["", "## 判断の範囲", "", "学習・検証とテストの初期状態を分離し、各保存点・全分岐を親軌跡単位で管理する。task06・09はコントローラの学習・検証から除外。人手の分解・テンプレートは共有しており、自動計画の研究とは呼ばない。oracleは特権的な参考条件。", "",
        "単一の学習設定・評価ノイズseed、9タスク・18初期状態の探索的な結果。途中の分母差を混ぜて優位を主張しない。効果予測が単純な達成確認を上回らなければ、その追加価値を支持しない。図の生成は優先しない。", "",
        "親軌跡完了: "+str(summary["parents_complete"])+"、分岐arm完了: "+str(summary["paired_arms_complete"])+"。",
        "\n入力・制御予算・split・閾値・期限は親ディレクトリのmanifest.json、モデル適合データはmodels/*_report.json、個別成否と切替はepisodes.csvに保存。"]
    (analysis/"report.md").write_text("\n".join(lines)+"\n")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args();summary=summarize(args.root)
    print(json.dumps({k:summary[k] for k in ("updated_utc","parents_complete","paired_arms_complete","completed_episodes","complete_five_condition_initial_states")}))
