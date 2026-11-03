"""Read-only result aggregation and explicit decisions for the rescue pilot."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from effect_worker import write
from rescue_protocol import MAIN_CASES,main_decision,pair_table,read_rows,window_phenomena
from rescue_runtime import last_contact_reference


LABELS=dict(contact_stall="接触後の停滞",contact_missing="接触未成立",target_unreached="対象への未到達",
    achieved_goal_regression="達成済み目標の崩壊",oscillation_or_reversal="振動・後退",
    terminal_progress="終盤も進展",unresolved="未確定")


def phase_rows(root,phase):
    rows=defaultdict(dict)
    for path in sorted((root/phase).glob("*/result.json")):
        data=json.loads(path.read_text());job=data["job"]
        rows[job["case"],job["seed"]][job["condition"]]=data
    return rows


def failure_classification(root):
    cases=[];candidates=defaultdict(list)
    for path in sorted((root/"replay").glob("*/result.json")):
        info=json.loads(path.read_text())
        if not info["replay_matches"]:
            cases.append(dict(case=info["case"],group=info["group"],original_success=info["original_success"],labels=["unresolved"],replay_matches=False));continue
        trace=read_rows(path.parent/"trace.jsonl")
        # Only the terminal failure window labels a failure. Earlier transient
        # approach/contact absence is recorded as evidence, not a failure cause.
        observed,evidence=window_phenomena(trace)
        eef=np.asarray([row["eef"] for row in trace[-50:]],float)
        evidence.update(window_start=trace[-50]["step"],window_end=trace[-1]["step"],
            eef_path_mm=float(np.linalg.norm(np.diff(eef,axis=0),axis=1).sum()*1000),
            eef_displacement_mm=float(np.linalg.norm(eef[-1]-eef[0])*1000))
        for i in range(2):
            if any(row["truth"]["raw_goals"][i] for row in trace) and not any(row["truth"]["raw_goals"][i] for row in trace[-5:]):
                observed.append("achieved_goal_regression")
        if trace[-1]["normalized_goal_distance"] is not None and trace[-50]["normalized_goal_distance"] is not None and trace[-1]["target"]==trace[-50]["target"]:
            reduction=trace[-50]["normalized_goal_distance"]-trace[-1]["normalized_goal_distance"]
            evidence["terminal_normalized_goal_distance_reduction"]=reduction
            if reduction>.01:
                observed.append("terminal_progress")
        record=dict(case=info["case"],group=info["group"],original_success=info["original_success"],
            replay_matches=True,labels=sorted(set(observed)) or ["unresolved"],terminal_evidence=evidence,
            final_truth=info["final_truth"],raw_scan_labels=info["labels"])
        cases.append(record)
        if info["original_success"]:
            continue
        for label in ("contact_stall","contact_missing","oscillation_or_reversal"):
            if label not in record["labels"]:
                continue
            for end in range(300,len(trace)-4,5):
                labels,numbers=window_phenomena(trace[:end])
                if label not in labels:
                    continue
                reference=last_contact_reference(trace[:end])
                if reference is not None:
                    candidates[label].append(dict(case=info["case"],step=end,label=label,
                        teacher_reference=reference,evidence=numbers));break
    priority={"contact_stall":0,"contact_missing":1,"oscillation_or_reversal":2}
    eligible=[label for label,values in candidates.items() if len(values)>=2]
    eligible.sort(key=lambda label:(-len(candidates[label]),priority[label]))
    chosen=eligible[0] if eligible else None
    selection=sorted(candidates[chosen],key=lambda row:row["case"])[:4] if chosen else []
    result=dict(cases=cases,chosen_physical_phenomenon=chosen,physical_selection=selection,
                rule="terminal 50-control phenotype; earliest matching window at/after300 with observed prior target contact; no outcome-based repair search")
    write(root/"classification.json",result)
    return result


def summarize(root):
    root=Path(root);(root/"analysis").mkdir(exist_ok=True)
    classification=failure_classification(root)
    main=phase_rows(root,"main")
    meta=json.loads((root/"replay_cases.json").read_text()) if (root/"replay_cases.json").exists() else {}
    original={case:value["original_success"] for case,value in meta.items()}
    eligible=[case for case in MAIN_CASES if (root/"prompts"/(case+".json")).exists()]
    decision=main_decision(main,original,eligible) if original else dict(complete=False,passes=False,reason="replay pending")
    physical=phase_rows(root,"physical");confirmation=phase_rows(root,"confirmation")
    physical_tables={arm:pair_table(physical,arm,"baseline5") for arm in ("replan2","teacher")}
    confirmation_tables={"C_vs_A":pair_table(confirmation,"C","A"),"C_vs_D":pair_table(confirmation,"C","D")}
    counts={phase:len(list((root/phase).glob("*/result.json"))) for phase in ("main","physical","confirmation")}
    incomplete=[]
    for phase in ("main","physical","confirmation"):
        incomplete.extend(str(path.parent.relative_to(root)) for path in (root/phase).glob("*/job.json") if not (path.parent/"result.json").exists())
    final=(root/"PIPELINE_FINISHED.json").exists()
    if not final:
        adoption="収集中。途中の成功率で手法を選ばない。"
    elif decision["passes"]:
        positive=all(value["n_seed_pairs"]==8 and value["net"]>0 for value in confirmation_tables.values()) and confirmation_tables["C_vs_A"]["harm"]==0
        adoption="全体目標を保持した現在操作指定を、次段階の候補に残す。切替は特権情報を使っており、実行可能な方式の完成ではない。" if positive else "主比較の予備基準は通過したが、未使用状態での追加価値を確認できず、今回は候補を採用しない。"
    else:
        teacher=physical_tables["teacher"];short=physical_tables["replan2"]
        selected=json.loads((root/"PHYSICAL_SELECTION.json").read_text())["states"] if (root/"PHYSICAL_SELECTION.json").exists() else []
        expected_teacher=2*sum(row["teacher_available"] for row in selected)
        expected_short=2*len(selected)
        def robust_rescues(table):
            return [case for case,delta in table["parent_mean_deltas"].items() if delta>0]
        if expected_teacher and teacher["n_seed_pairs"]==expected_teacher and teacher["net"]>0 and teacher["harm"]==0 and len(robust_rescues(teacher))>=2:
            adoption="教師による短い修正で複数の元方策失敗が復帰した。対象の観測現象と修正動作を固定し、通常成功データを混ぜた修正学習を次段階の候補にする。教師そのものを新手法とは呼ばない。"
        elif expected_short and short["n_seed_pairs"]==expected_short and short["net"]>0 and short["harm"]==0:
            adoption="短い再計画周期を強いベースラインとして残す。追加推論コストがあり、新手法とは呼ばない。"
        elif not decision.get("complete") and not physical:
            adoption="必要な対応比較が未完了のため、採否は保留する。未完了を失敗に補完しない。"
        else:
            adoption="元方策に対する正味改善を支持する結果は得られていない。候補を採用せず、予測器や指示文を追加して方向を維持しない。"
    error=root/"PIPELINE_ERROR.json"
    if error.exists():
        adoption="実行パイプラインが技術的なエラーで停止。未完了を方策失敗や方法の無効性に読み替えず、採否は保留する。"
    auditfile=root/"analysis/data_audit.json"
    if auditfile.exists() and not json.loads(auditfile.read_text())["passed"]:
        adoption="データ整合性監査に不一致があるため、方法の採否は保留する。"
    payload=dict(updated_utc=datetime.now(timezone.utc).isoformat(),counts=counts,incomplete=incomplete,
        main_decision=decision,physical=physical_tables,confirmation=confirmation_tables,classification=classification,
        final=final,adoption=adoption)
    write(root/"analysis/summary.json",payload)
    lines=["# SkillNetの元の失敗を救済する半日診断", "", adoption, "",
        f"完了：主比較 {counts['main']}/48、物理診断 {counts['physical']}/24、未使用状態確認 {counts['confirmation']}/24。後二つは条件分岐で、両方を実施する計画ではない。", "",
        "## 元方策の8失敗：再生による観測分類", "", "接触解析は画像・自己状態の再現が一致した例のみ。末尾50制御ステップの現象と達成状態の崩れを分類し、原因は断定しない。", "",
        "| 例 | 元方策/正解切替の群 | 再現一致 | 観測現象 | 末尾接触step/50 | 正規化変動幅 |", "|---|---|---|---|---:|---:|"]
    for row in classification["cases"]:
        if not row["original_success"]:
            evidence=row.get("terminal_evidence",{})
            lines.append(f"| {row['case']} | {row['group']} | {row['replay_matches']} | {'、'.join(LABELS[k] for k in row['labels'])} | {evidence.get('target_contact_steps','—')} | {evidence.get('normalized_progress_span','—')} |")
    lines += ["", "## 指示の4条件比較", "", "A=元指示、B=残り指示、C=全体指示＋現在操作、D=全体指示＋中立的リマインダ。切替後固定・巻戻しなし。選択した6状態の開発用診断であり、一般的な成功率とは呼ばない。", "",
        "| 比較 | seed対応組 | 保存状態 | 救済 | 悪化 | 正味 |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("C_vs_A","C_vs_D","C_vs_B","D_vs_A"):
        if name in decision:
            row=decision[name];lines.append(f"| {name} | {row['n_seed_pairs']} | {row['n_parent_states']} | {row['rescue']} | {row['harm']} | {row['net']:+d} |")
    lines += ["", "予備的な進行基準通過："+str(decision["passes"]), "", "## 次段階の比較", "", "| 比較 | seed対応組 | 救済 | 悪化 | 正味 |", "|---|---:|---:|---:|---:|"]
    for name,row in {**physical_tables,**confirmation_tables}.items():
        if row["n_seed_pairs"]:
            lines.append(f"| {name} | {row['n_seed_pairs']} | {row['rescue']} | {row['harm']} | {row['net']:+d} |")
    selection_file=root/"PHYSICAL_SELECTION.json"
    if selection_file.exists():
        selection=json.loads(selection_file.read_text())
        lines += ["", "物理診断の対象現象："+str(selection["phenomenon"])+"。"]
        for row in selection["states"]:
            if not row["teacher_available"]:
                lines.append(row["case"]+"の教師条件は未実施："+row["teacher_unavailable_reason"]+"。失敗に数えない。")
    lines += ["", "未完了："+str(len(incomplete))+"。失敗に補完しない。", "",
        "再計画を短くした条件は推論回数・実時間が増える。教師は実行した最大20ステップも800の総予算に含める。正解状態・接触は診断専用で、実行可能な方式の入力とは区別する。", "",
        "新しい学習、DMP、expert追加は実施しない。全体目標と現在操作の併用や失敗修正学習自体を新規性とはしない。"]
    if error.exists():
        lines += ["", "**実行エラー：** "+error.read_text()+"。方策の失敗とは数えない。"]
    (root/"analysis/report.md").write_text("\n".join(lines)+"\n")
    return payload
