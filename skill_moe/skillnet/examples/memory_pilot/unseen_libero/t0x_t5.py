#!/usr/bin/env python3
"""T5: calibrated top-10 motion disagreement and Expert-generated gripper closed loop."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import kq2cl_common as C
import kq2cl_closed as K
import t0x_closed as T1
import t0x_report as R
from t0x_common import OUT, ROOT, calibrate_disagreement, progress, topk_disagreement, t0_add, t0_from_disagreement, write_json

T5 = OUT / "t5"
CAL_PATH = T5 / "calibration.json"
BASE_CL_OUT = C.OUT
REAL_POPEN = subprocess.Popen
CONDITIONS = ("AaddGexp", "AtopkGexp", "AtopkGdec")


def t5log(message: str, **fields):
    progress("t5", message, **fields)
    T5.mkdir(parents=True, exist_ok=True)
    with (T5 / "progress.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"jst": __import__("t0x_common").timestamp(), "message": message, **fields},
                           ensure_ascii=False, default=float) + "\n")


def calibration_from_dev() -> dict:
    """Get D from the fixed Dev-v2 retrieval cache; never inspect either Eval suite here."""
    if CAL_PATH.exists():
        return json.loads(CAL_PATH.read_text(encoding="utf-8"))
    T5.mkdir(parents=True, exist_ok=True)
    retrieval = np.load(BASE_CL_OUT / "retrieval/iii__dev2.npz")
    pred = np.load(BASE_CL_OUT / "heads/pred_iii.npz")
    anchor = json.loads((OUT / "t4/calibration.json").read_text(encoding="utf-8"))
    ids = np.asarray(retrieval["ids"], np.int64)[:, :10]
    logits = np.asarray(retrieval["logits"], np.float64)[:, :10]
    if ids.shape[0] != len(pred["dev2__ehat"]):
        raise AssertionError("Dev retrieval and confidence-head rows differ")
    all_units = np.load(C.ROOT / "key-motion-01/final/data/bank_actions.npy", mmap_mode="r")
    actions = np.asarray(all_units[ids, :, :6], dtype=np.float32)
    d, probs = topk_disagreement(actions, logits, k=10)
    ehat = np.asarray(pred["dev2__ehat"], np.float64)
    b = float(anchor["b"])
    cal = calibrate_disagreement(d, ehat, b)
    true_error = np.asarray(retrieval["err"], np.float64)
    rho = float(spearmanr(d, true_error).statistic)
    rho_e = float(spearmanr(ehat, true_error).statistic)
    cal.update({"source": "KQ2-CL iii Dev-v2 retrieval top-10 and fixed confidence head",
                "dev_rows": int(len(d)), "k": 10, "score_softmax": "softmax(top10_logits), renormalized within top10",
                "arm_mse": "mean over 10 time steps x 6 arm dimensions, stored actions unchanged",
                "ehat_fallback_rate": cal["p_b"],
                "realized_D_fallback_rate_strict_gt": float(np.mean(d > cal["D_b"])),
                "D_spearman_vs_top1_true_arm_error": rho,
                "ehat_spearman_vs_top1_true_arm_error": rho_e,
                "D_quantiles_p05_p25_p50_p75_p95": np.quantile(d, [.05, .25, .5, .75, .95]).tolist(),
                "ehat_quantiles_p05_p25_p50_p75_p95": np.quantile(ehat, [.05, .25, .5, .75, .95]).tolist(),
                "fallback_b": b,
                "retrieval_cache": str(BASE_CL_OUT / "retrieval/iii__dev2.npz"),
                "confidence_cache": str(BASE_CL_OUT / "heads/pred_iii.npz")})
    np.savez_compressed(T5 / "dev_calibration_arrays.npz", D=d.astype(np.float32), ehat=ehat.astype(np.float32),
                        true_top1_error=true_error.astype(np.float32), top10_ids=ids.astype(np.int32),
                        top10_probabilities=probs.astype(np.float32))
    write_json(CAL_PATH, cal)
    t5log("Dev-only D calibration complete", rows=len(d), D_b=cal["D_b"], p_b=cal["p_b"],
          spearman=cal["D_spearman_vs_top1_true_arm_error"])
    return cal


def build_specs(cal: dict) -> dict:
    base = json.loads((OUT / "t1/conditions-libero90.json").read_text(encoding="utf-8"))["Aadd"]
    b = float(base["tpol"]["b"])
    if abs(b - float(cal["b"])) > 1e-10:
        raise AssertionError("T5 threshold differs from the T1 iii confidence head")
    dpol = {"formula": "topk_disagreement", "confidence_mode": "disagreement",
            "gripper_mode": "expert", "condition": "AtopkGexp", "b": b,
            "c": (.35**2 - .15**2) / b, "lo": .15, "hi": .35, "calibration": str(CAL_PATH)}
    out = {"AaddGexp": base | {"tpol": base["tpol"] | {"formula": "additive_radius",
                                                           "confidence_mode": "ehat",
                                                           "gripper_mode": "expert",
                                                           "condition": "AaddGexp"}},
           "AtopkGexp": base | {"tpol": dpol},
           "AtopkGdec": base | {"tpol": dpol | {"gripper_mode": "decoder", "condition": "AtopkGdec"}}}
    return out


def _popen_with_t5_server(args, *pos, **kw):
    if isinstance(args, (list, tuple)):
        args = list(args)
        for i, v in enumerate(args):
            if str(v).endswith("/kq2cl_serve.py") or str(v) == str(C.HERE / "kq2cl_serve.py"):
                args[i] = str(C.HERE / "t0x_t5_serve.py")
                break
    return REAL_POPEN(args, *pos, **kw)


def run_suite(suite: str, condition: str, *, smoke: bool, workers: int, fraction: float, cpus: str):
    if suite not in ("libero90", "libero10") or condition not in CONDITIONS:
        raise ValueError((suite, condition))
    C.OUT = T5 / ("smoke" if smoke else "closed-loop")
    C.OUT.mkdir(parents=True, exist_ok=True)
    stage = C.OUT
    spec = build_specs(json.loads(CAL_PATH.read_text(encoding="utf-8")))
    spec_path = stage / f"conditions-{suite}.json"
    if spec_path.exists() and json.loads(spec_path.read_text(encoding="utf-8")) != spec:
        raise RuntimeError(f"refusing to overwrite a different T5 condition spec: {spec_path}")
    if not spec_path.exists():
        write_json(spec_path, spec)
    K.PLAN[suite] = dict(K.PLAN[suite])
    K.PLAN[suite][condition] = range(10)
    old_popen = K.subprocess.Popen
    K.subprocess.Popen = _popen_with_t5_server
    prior = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    try:
        t5log("condition start", suite=suite, condition=condition, smoke=smoke, gpu=K.gpu_used(), fraction=fraction)
        K.run([condition], workers=1 if smoke else workers, fraction=fraction, smoke=smoke,
              cpus=[cpus], spec_path=str(spec_path), suite=suite)
        root = stage / ("closed-loop-smoke" + ("" if suite == "libero90" else "-libero10")
                        if smoke else f"closed-loop/{suite}")
        rows = K.read_trials(root / condition)
        expected = 2 if smoke else 100
        complete = sum(x.get("status") == "complete" for x in rows.values())
        if complete != expected:
            raise RuntimeError(f"{suite}/{condition}: {complete}/{expected} complete records")
        t5log("condition complete", suite=suite, condition=condition, smoke=smoke, completed=complete,
              successes=sum(bool(x["success"]) for x in rows.values() if x.get("status") == "complete"))
    finally:
        K.subprocess.Popen = old_popen
        for key, value in prior.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value
        C.OUT = BASE_CL_OUT


def _switches_for_trial(row):
    executed = []
    for q in row.get("selected_units_and_t0", []):
        executed.extend(np.sign(np.asarray(q.get("expert_gripper", q.get("gripper_chunk", [])), float)[:5]).tolist())
    return int(np.sum(np.asarray(executed[1:]) != np.asarray(executed[:-1]))) if len(executed) > 1 else 0


def summarize_condition(suite: str, condition: str, refs: dict) -> dict:
    # KQ2-CL's runner appends its own "closed-loop/<suite>" under C.OUT.
    path = T5 / "closed-loop" / "closed-loop" / suite / condition / "trials.jsonl"
    rows = R.by_trial(R.read_jsonl(path))
    records = list(rows.values())
    tasks, _ = R.baseline_for_suite(suite)
    qs = [q for row in records for q in row.get("selected_units_and_t0", []) if isinstance(q, dict)]
    sw = {"success": [], "failure": []}
    for row in records:
        sw["success" if row.get("success") else "failure"].append(_switches_for_trial(row))
    def qstat(vals):
        a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
        return {} if not len(a) else {"p05": float(np.quantile(a, .05)), "median": float(np.median(a)),
                                      "p95": float(np.quantile(a, .95)), "mean": float(a.mean())}
    success_count = sum(bool(x["success"]) for x in records)
    result = {"completed": len(records), "successes": success_count,
              "wilson95": R.wilson(success_count, len(records)),
              "tasks": {str(t["task_id"]): {"name": t["name"], "successes": sum(bool(rows.get((t["task_id"], i), {}).get("success")) for i in range(10)),
                                               "completed": sum((t["task_id"], i) in rows for i in range(10))} for t in tasks},
              "mcnemar": {key: R.mcnemar(rows, ref) for key, ref in refs.items()},
              "query_diagnostics": {"queries": len(qs),
                                    "D": qstat([q.get("disagreement_D") for q in qs]),
                                    "t0": qstat([q.get("t0") for q in qs]),
                                    "ehat": qstat([q.get("confidence_error") for q in qs]),
                                    "fallback_rate": float(np.mean([bool(q.get("fallback_pi05")) for q in qs])) if qs else None,
                                    "gripper_mode": sorted(set(q.get("gripper_mode", "unknown") for q in qs))},
              "gripper_switches_per_trial": {k: qstat(v) | {"trials": len(v)} for k, v in sw.items()},
              "median_success_steps": float(np.median([r["env_steps"] for r in records if r.get("success")])) if success_count else None}
    return result


def build_results() -> dict:
    t4cal = json.loads((OUT / "t4/calibration.json").read_text(encoding="utf-8"))
    result = {"created_jst": __import__("t0x_common").timestamp(), "calibration": json.loads(CAL_PATH.read_text()),
              "conditions": build_specs(json.loads(CAL_PATH.read_text())), "suites": {},
              "smoke": {}, "oom_log_matches": {}}
    for suite in ("libero90", "libero10"):
        refs = {"Aadd": R.by_trial(R.read_jsonl(OUT / "t1/closed-loop" / suite / "Aadd/trials.jsonl")),
                "Aconst": R.by_trial(R.read_jsonl(ROOT / "kq2-cl-01/closed-loop" / suite / "Aconst/trials.jsonl")),
                "pi05": R.baseline_for_suite(suite)[1]}
        if suite == "libero10":
            refs["AconstPG"] = R.by_trial(R.read_jsonl(OUT / "t3/closed-loop/libero10/AconstPG/trials.jsonl"))
        result["suites"][suite] = {}
        for cond in CONDITIONS:
            result["suites"][suite][cond] = summarize_condition(suite, cond, refs)
        result["oom_log_matches"][suite] = 0
        for log in (T5 / "closed-loop" / "closed-loop" / suite / "servers").glob("*.log"):
            text = log.read_text(errors="replace").lower()
            result["oom_log_matches"][suite] += sum(text.count(x) for x in ("out of memory", "resource_exhausted", "cuda_error_out_of_memory"))
    for cond in CONDITIONS:
        rows = R.read_jsonl(T5 / "smoke/closed-loop-smoke" / cond / "trials.jsonl")
        result["smoke"][cond] = {"completed": sum(r.get("status") == "complete" for r in rows),
                                  "successes": sum(r.get("status") == "complete" and r.get("success") for r in rows)}
    result["status"] = {s: {c: x["completed"] for c, x in cs.items()} for s, cs in result["suites"].items()}
    return result


def markdown(result: dict) -> str:
    cal = result["calibration"]
    lines = ["## T5：上位候補の運動不一致による信頼度と Expert グリッパー", "",
             f"Dev v2 のみで校正（{cal['dev_rows']}窓）。ê のフォールバック割合 p_b={cal['p_b']:.3f} に合わせた D の上側しきい値は D_b={cal['D_b']:.6g}（超過時に π0.5）。D と正解の top-1 腕誤差の Dev Spearman={cal['D_spearman_vs_top1_true_arm_error']:.3f}、ê の Spearman={cal['ehat_spearman_vs_top1_true_arm_error']:.3f}。T1 と同じ K-x1/(iii)/K1:500 を使い、学習はしていない。G-exp は hold-last を下書きに入れ、G-dec はデコーダ出力を下書きに入れ、両方とも bridge 後の Expert グリッパー出力をそのまま返す。", "",
             "| Suite | 条件 | 成功/完了 | Wilson 95% | π0.5 p | A-const p | A-add p | A-const-pg p | fallback | D P05/50/95 | t0 P05/50/95 | 切替/試行 成功 | 切替/試行 失敗 |", "|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|"]
    for suite in ("libero90", "libero10"):
        for cond, x in result["suites"][suite].items():
            q=x["query_diagnostics"]
            trip=lambda z: "—" if not z else f"{z['p05']:.3f}/{z['median']:.3f}/{z['p95']:.3f}"
            pp=lambda name: f"{x['mcnemar'][name]['p_exact']:.3g}" if name in x["mcnemar"] else "—"
            lines.append(f"| {suite} | {cond} | {x['successes']}/{x['completed']} | [{x['wilson95'][0]:.3f}, {x['wilson95'][1]:.3f}] | {pp('pi05')} | {pp('Aconst')} | {pp('Aadd')} | {pp('AconstPG')} | {100*q['fallback_rate']:.1f}% | {trip(q['D'])} | {trip(q['t0'])} | {trip(x['gripper_switches_per_trial']['success'])} | {trip(x['gripper_switches_per_trial']['failure'])} |")
    lines += ["", "T5 条件は試行番号・初期状態・seed_base=7、wait=10、replan=5、最大520制御stepを T1 と共有する。McNemar は同一 task/trial の対応試行で計算。グリッパー切替数は実行した各チャンクの先頭5指令だけを連結して数えた。", "", "**タスク別成功数**", "", "| Suite | 条件 | Task | 成功/完了 |", "|---|---|---|---:|"]
    for suite, conds in result["suites"].items():
        for cond, x in conds.items():
            for tid, task in x["tasks"].items():
                lines.append(f"| {suite} | {cond} | {task['name']} (`{tid}`) | {task['successes']}/{task['completed']} |")
    lines += ["", f"Dev 校正: D P05/P25/P50/P75/P95 = {'/'.join(f'{x:.4g}' for x in cal['D_quantiles_p05_p25_p50_p75_p95'])}; D_b={cal['D_b']:.6g}; p_b={cal['p_b']:.3f}。D は上位10候補内の softmax(検索logit)重み付き top-1 との arm6 MSE。", "", f"スモーク: " + ", ".join(f"{c} {v['successes']}/{v['completed']}" for c,v in result['smoke'].items()) + ". OOM ログ検出数: " + json.dumps(result['oom_log_matches']) + ".", "", "JST の進捗・実行ログは `t5/progress.log` と `t5/t5-run.log`。", ""]
    return "\n".join(lines)


def run_all(*, workers=6, fraction=.20, server_cpus="24-39"):
    cal = calibration_from_dev()
    # Smoke each condition first, on the same public task and two initial states.
    for cond in CONDITIONS:
        run_suite("libero90", cond, smoke=True, workers=1, fraction=fraction, cpus=server_cpus)
    # Sequential GPU ownership: finish each 100-trial condition before starting the next server.
    for suite in ("libero90", "libero10"):
        for cond in CONDITIONS:
            run_suite(suite, cond, smoke=False, workers=workers, fraction=fraction, cpus=server_cpus)
    result = build_results()
    write_json(T5 / "results.json", result)
    (T5 / "report.md").write_text(markdown(result), encoding="utf-8")
    root_report = OUT / "report.md"
    old = root_report.read_text(encoding="utf-8") if root_report.exists() else ""
    old = old.split("\n## T5：", 1)[0].rstrip()
    root_report.write_text(old + "\n\n" + markdown(result), encoding="utf-8")
    (T5 / "DONE").write_text(f"DONE {__import__('t0x_common').timestamp()} JST\n", encoding="utf-8")
    t5log("DONE", trials={s: {c: x["completed"] for c,x in cs.items()} for s,cs in result["suites"].items()},
          oom_log_matches=result["oom_log_matches"])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("calibrate", "smoke", "run", "all"))
    ap.add_argument("--suite", choices=("libero90", "libero10"), default="libero90")
    ap.add_argument("--condition", choices=CONDITIONS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fraction", type=float, default=.20)
    ap.add_argument("--server-cpus", default="24-39")
    args = ap.parse_args()
    os.environ.setdefault("SKILLNET_SUBTASK_GENERATE", "0")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.command == "calibrate":
        print(json.dumps(calibration_from_dev(), indent=2))
    elif args.command == "all":
        run_all(workers=args.workers, fraction=args.fraction, server_cpus=args.server_cpus)
    else:
        calibration_from_dev()
        if not args.condition:
            raise SystemExit("--condition is required for smoke/run")
        run_suite(args.suite, args.condition, smoke=args.command == "smoke", workers=args.workers,
                  fraction=args.fraction, cpus=args.server_cpus)


if __name__ == "__main__":
    main()
