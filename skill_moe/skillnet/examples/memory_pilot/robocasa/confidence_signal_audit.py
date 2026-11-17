"""Offline confidence-signal audit for RCX3-A.

Reads RCX2 caches and closed-loop logs without changing the serving policy.
All fitted confidence mappings and thresholds use the 741-row Dev split only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from scipy.stats import binomtest

import numpy as np
from scipy.stats import rankdata, spearmanr


ROOT = Path("/work/.skillnet-state/outputs/robocasa-memory/latent-skill-01")
OUT = ROOT / "conf"
MODES = ("p5", "e_vlm_h")


def corr(x, y):
    x, y = np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3 or np.std(x[good]) == 0 or np.std(y[good]) == 0:
        return float("nan")
    return float(spearmanr(x[good], y[good]).statistic)


def auc(labels, scores):
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, float)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    ranks = rankdata(scores)
    n1, n0 = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def softmax_entropy(scores, scale):
    z = np.asarray(scores, np.float64) * float(scale)
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return -(p * np.log(np.maximum(p, 1e-30))).sum(axis=-1)


def _scores_and_features(mode, units, rows, drafts, split):
    score_path = ROOT / f"key/eval/{mode}_{split}_scores_fp16.npy"
    top1_path = ROOT / f"key/eval/{mode}_{split}_top1.npy"
    scores = np.load(score_path, mmap_mode="r")
    selected = np.load(top1_path).astype(np.int64)
    if scores.shape[0] != len(rows["task"]) or selected.shape != (scores.shape[0],):
        raise ValueError(f"Dev score rows mismatch: {mode}")
    top10 = np.argpartition(np.asarray(scores), -10, axis=1)[:, -10:]
    topvals = np.take_along_axis(np.asarray(scores), top10, axis=1)
    order = np.argsort(-topvals, axis=1)
    top10 = np.take_along_axis(top10, order, axis=1)
    topvals = np.take_along_axis(topvals, order, axis=1)
    # Preserve the float32 ranking's saved winner and avoid duplicating it if fp16 ties reorder it.
    for i, winner in enumerate(selected):
        remainder = top10[i][top10[i] != winner]
        top10[i] = np.r_[winner, remainder[:9]]
    topvals = np.asarray(scores)[np.arange(len(selected))[:, None], top10]

    cfg = json.loads((ROOT / "key/selected_configs.json").read_text())["selected"][mode]
    state = __import__("torch").load(ROOT / "key/runs" / cfg["run"] / "model.pt", map_location="cpu", weights_only=False)
    scale = float(state["logit_scale"].exp().item())
    coeff = units["coeff"]
    topcoeff = coeff[top10]
    wincoeff = coeff[selected]
    spread = np.sqrt(np.mean((topcoeff - wincoeff[:, None]) ** 2, axis=(1, 2, 3)))
    unit_distance = np.sqrt(np.mean((coeff[np.load(ROOT / f"expert/drafts/{split}_{mode}.npz")["first"].astype(int)] -
                                     coeff[np.load(ROOT / f"expert/drafts/{split}_{'e_vlm_h' if mode == 'p5' else 'p5'}.npz")["first"].astype(int)]) ** 2,
                                    axis=(1, 2)))
    draft = drafts["draft"][:, :10]
    history = rows["history"]
    hmask = rows["history_mask"]
    selected_units = drafts["first"].astype(int)
    order = np.lexsort((rows["frame"], rows["episode"], rows["task"]))
    same_unit = np.zeros(len(selected_units), np.float32)
    for j in range(1, len(order)):
        prev_i, cur_i = order[j - 1], order[j]
        if rows["task"][prev_i] == rows["task"][cur_i] and rows["episode"][prev_i] == rows["episode"][cur_i]:
            same_unit[cur_i] = float(selected_units[prev_i] == selected_units[cur_i])
    last = np.zeros((len(history), 7), np.float32)
    for i in range(len(history)):
        valid = np.flatnonzero(hmask[i])
        if valid.size:
            last[i] = history[i, valid[-1]]
    jump = np.sqrt(np.mean((draft[:, 0, :6] - last[:, :6]) ** 2, axis=1))
    return {
        "k_top1": np.asarray(scores)[np.arange(len(selected)), selected].astype(float),
        "k_gap": (topvals[:, 0] - topvals[:, 1]).astype(float),
        "k_entropy10": softmax_entropy(topvals, scale),
        "r_spread10": spread,
        "r_p5_vs_e_top1": unit_distance,
        "c_draft_jump": jump,
        "c_same_unit_repeat": same_unit,
        "selected": selected,
        "top10": top10,
        "scale": scale,
    }


def _expert_error_and_grid(mode, per, first10, split):
    dist = per[f"draft__{mode}__dist_pi05"].astype(float)
    selector = json.loads((ROOT / f"closed-loop/selector_{mode}.json").read_text())
    grid = np.asarray(selector["t0_grid"], float)
    w = np.asarray(selector["W"], float)
    x = np.stack([np.ones_like(dist), np.log(np.maximum(dist, 1e-6))], axis=1)
    estimates = x @ w
    idx = estimates.argmin(axis=1)
    post = per[f"expert__{mode}__arm10"][np.arange(len(idx)), idx].astype(float)
    return post, grid[idx], first10


def dev_audit():
    rows = dict(np.load(ROOT / "rows/dev.npz"))
    units = dict(np.load(ROOT / "units/memory_units.npz"))
    per = dict(np.load(ROOT / "expert/eval/dev/per_row.npz"))
    first10 = dict(np.load(ROOT / "expert/eval/dev/first10.npz"))
    drafts = {m: dict(np.load(ROOT / f"expert/drafts/dev_{m}.npz")) for m in MODES}
    feature_sets = {m: _scores_and_features(m, units, rows, drafts[m], "dev") for m in MODES}
    result = {"n_dev": len(rows["task"]), "modes": {}, "a_b_candidates": {}}
    all_predictors = {}
    for mode in MODES:
        f = feature_sets[mode]
        draft_error = per[f"draft__{mode}__arm10"].astype(float)
        post_error, selected_t0, _ = _expert_error_and_grid(mode, per, first10, "dev")
        refs = {
            "k_top1": f["k_top1"], "k_gap": f["k_gap"],
            "k_entropy10": f["k_entropy10"], "r_spread10": f["r_spread10"],
            "r_p5_vs_e_top1": f["r_p5_vs_e_top1"], "c_draft_jump": f["c_draft_jump"],
            "c_same_unit_repeat": f["c_same_unit_repeat"],
            "pi05_dist_reference_only": per[f"draft__{mode}__dist_pi05"],
        }
        mode_out = {"logit_scale": f["scale"], "draft_error_spearman": {}, "expert_error_spearman": {},
                    "features": {k: np.asarray(v).astype(float).tolist() for k, v in refs.items()}}
        for name, values in refs.items():
            mode_out["draft_error_spearman"][name] = corr(values, draft_error)
            mode_out["expert_error_spearman"][name] = corr(values, post_error)
        # Expert self-consistency, derived from existing cached generated actions only.
        low = first10[f"expert__{mode}__0.2"].astype(np.float32)[..., :6]
        high = first10[f"expert__{mode}__0.8"].astype(np.float32)[..., :6]
        f["s_t02_vs_t08_action_rms"] = np.sqrt(np.mean((low - high) ** 2, axis=(1, 2)))
        mode_out["features"]["s_t02_vs_t08_action_rms"] = f["s_t02_vs_t08_action_rms"].astype(float).tolist()
        mode_out["draft_error_spearman"]["s_t02_vs_t08_action_rms"] = corr(mode_out["features"]["s_t02_vs_t08_action_rms"], draft_error)
        mode_out["expert_error_spearman"]["s_t02_vs_t08_action_rms"] = corr(mode_out["features"]["s_t02_vs_t08_action_rms"], post_error)
        refs["s_t02_vs_t08_action_rms"] = np.asarray(mode_out["features"]["s_t02_vs_t08_action_rms"])
        all_predictors[mode] = ["k_top1", "k_gap", "k_entropy10", "r_spread10", "r_p5_vs_e_top1", "c_draft_jump", "c_same_unit_repeat", "s_t02_vs_t08_action_rms"]
        # Report spread of predicted errors without exposing target labels at inference.
        result["modes"][mode] = mode_out
        # Threshold policy: fit predicted draft error on Dev, then express a < b in error units.
        X = np.column_stack([refs[k] for k in all_predictors[mode]]).astype(float)
        med = np.nanmedian(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd[sd < 1e-8] = 1.0
        Z = (X - med) / sd
        Z = np.nan_to_num(Z)
        y = np.log(np.maximum(draft_error, 1e-6))
        # Small fixed ridge stabilizer; linear model is fitted on Dev only.
        beta = np.linalg.solve(Z.T @ Z + 1e-2 * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
        pred_logerr = Z @ beta + y.mean()
        mode_out["loglinear_composite"] = {"features": all_predictors[mode], "spearman_draft": corr(pred_logerr, draft_error),
                                             "spearman_expert": corr(pred_logerr, post_error),
                                             "coefficients": beta.tolist(), "center": med.tolist(), "scale": sd.tolist()}
        predicted_error = np.exp(pred_logerr)
        pi = per["pi05_arm10"].astype(float)
        exp_grid = per[f"expert__{mode}__arm10"].astype(float)
        grid = per["t0_grid"].astype(float)
        candidates = []
        for fallback_pct in (0, 10, 20, 30, 40):
            bq = np.quantile(predicted_error, 1 - fallback_pct / 100) if fallback_pct else np.nextafter(predicted_error.max(), np.inf)
            fallback = predicted_error >= bq
            best = None
            for aq in (10, 20, 30, 40, 50, 60):
                a = np.quantile(predicted_error, aq / 100)
                if a >= bq:
                    continue
                t = np.full(len(predicted_error), .1, float)
                ramp = (predicted_error > a) & (predicted_error < bq)
                t[ramp] = .1 + .8 * ((predicted_error[ramp] - a) / (bq - a))
                # Nearest cached t0 is an explicit discretization of the requested ramp.
                ti = np.abs(t[:, None] - grid[None, :]).argmin(axis=1)
                mse = np.where(fallback, pi, exp_grid[np.arange(len(t)), ti]).mean()
                rec = {"fallback_target_percentile": fallback_pct, "actual_fallback_rate": float(fallback.mean()),
                       "a_error_percentile": aq, "a_error_threshold": float(a),
                       "b_error_threshold": float(bq),
                       "dev_arm_mse": float(mse), "mean_t0_nonfallback": float(t[~fallback].mean()) if (~fallback).any() else None}
                if best is None or mse < best["dev_arm_mse"]:
                    best = rec
            candidates.append(best)
        result["a_b_candidates"][mode] = candidates
        result["modes"][mode]["confidence_fit"] = mode_out["loglinear_composite"]
    # P5-vs-E unit top1 and adjacent-window continuity, using no task label as a feature.
    p5 = drafts["p5"]["first"].astype(int)
    ev = drafts["e_vlm_h"]["first"].astype(int)
    order = np.lexsort((rows["frame"], rows["episode"], rows["task"]))
    same, adjacent = [], []
    for j in range(1, len(order)):
        i0, i1 = order[j - 1], order[j]
        if rows["task"][i0] == rows["task"][i1] and rows["episode"][i0] == rows["episode"][i1]:
            adjacent.append((i0, i1))
            same.append(float(p5[i0] == p5[i1] or ev[i0] == ev[i1]))
    result["continuity_same_selected_unit_rate"] = float(np.mean(same)) if same else None
    result["continuity_adjacent_dev_pairs"] = len(adjacent)
    result["notes"] = ["π0.5 距離は比較専用で、合成信号の学習入力から除外。", "Dev相関と閾値の当てはめは同じ741行内で行った記述的選択値。独立汎化推定ではない。", "t0≈.9 は Expert の下書き依存を弱め、π0.5生成に近づくため自己一致指標の解釈に注意。"]
    # One held-out confirmation pass: fixed Dev composite coefficients and logit scales.
    eval_rows = dict(np.load(ROOT / "rows/eval.npz"))
    eval_per = dict(np.load(ROOT / "expert/eval/eval/per_row.npz"))
    eval_first = dict(np.load(ROOT / "expert/eval/eval/first10.npz"))
    eval_drafts = {m: dict(np.load(ROOT / f"expert/drafts/eval_{m}.npz")) for m in MODES}
    result["eval_confirmation"] = {"n_eval": len(eval_rows["task"]), "modes": {}}
    for mode in MODES:
        ef = _scores_and_features(mode, units, eval_rows, eval_drafts[mode], "eval")
        draft_err = eval_per[f"draft__{mode}__arm10"].astype(float)
        post_err, _, _ = _expert_error_and_grid(mode, eval_per, eval_first, "eval")
        mode_result = {"draft_error_spearman": {}, "expert_error_spearman": {}}
        for name in ("k_top1", "k_gap", "k_entropy10", "r_spread10", "r_p5_vs_e_top1", "c_draft_jump", "c_same_unit_repeat"):
            mode_result["draft_error_spearman"][name] = corr(ef[name], draft_err)
            mode_result["expert_error_spearman"][name] = corr(ef[name], post_err)
        mode_result["draft_error_spearman"]["pi05_dist_reference_only"] = corr(eval_per[f"draft__{mode}__dist_pi05"], draft_err)
        mode_result["expert_error_spearman"]["pi05_dist_reference_only"] = corr(eval_per[f"draft__{mode}__dist_pi05"], post_err)
        low = eval_first[f"expert__{mode}__0.2"].astype(np.float32)[..., :6]
        high = eval_first[f"expert__{mode}__0.8"].astype(np.float32)[..., :6]
        s = np.sqrt(np.mean((low - high) ** 2, axis=(1, 2)))
        ef["s_t02_vs_t08_action_rms"] = s
        mode_result["draft_error_spearman"]["s_t02_vs_t08_action_rms"] = corr(s, draft_err)
        mode_result["expert_error_spearman"]["s_t02_vs_t08_action_rms"] = corr(s, post_err)
        spec = result["modes"][mode]["loglinear_composite"]
        names = spec["features"]
        X = np.column_stack([ef[k] for k in names]).astype(float)
        Z = np.nan_to_num((X - np.asarray(spec["center"])) / np.asarray(spec["scale"]))
        # Recover training target mean from the serialized linear predictor by refitting intercept from Dev.
        dev_mode = result["modes"][mode]
        dev_target = per[f"draft__{mode}__arm10"].astype(float)
        dev_X = np.column_stack([feature_sets[mode][k] for k in names]).astype(float)
        dev_Z = np.nan_to_num((dev_X - np.asarray(spec["center"])) / np.asarray(spec["scale"]))
        intercept = float(np.mean(np.log(np.maximum(dev_target, 1e-6)) - dev_Z @ np.asarray(spec["coefficients"])))
        pred_log = Z @ np.asarray(spec["coefficients"]) + intercept
        mode_result["loglinear_composite"] = {"draft_error_spearman": corr(pred_log, draft_err),
                                                 "expert_error_spearman": corr(pred_log, post_err)}
        result["eval_confirmation"]["modes"][mode] = mode_result
    return result


def load_json(path):
    return json.loads(Path(path).read_text())


def closed_log_audit():
    base_path = Path("/work/.skillnet-state/outputs/robocasa-memory/baseline-01/v6-nosubtask/eval/results.json")
    baseline = load_json(base_path)
    baseline_by = {(r["task"], r["trial"]): r for r in baseline["records"] if r.get("status") == "complete"}
    baseline_success_steps = np.asarray([r["steps"] for r in baseline_by.values() if r["success"]], float)
    out = {}
    for mode in ("p0", "p5", "e_vlm_h"):
        eval_path = ROOT / f"closed/{mode}/eval/results.json"
        evaldata = load_json(eval_path) if eval_path.exists() else {"records": []}
        by = {(r["task"], r["trial"]): r for r in evaldata["records"] if r.get("status") == "complete"}
        qpath = ROOT / f"closed/{mode}/queries.jsonl"
        qby = {}
        if qpath.exists():
            for line in qpath.read_text().splitlines():
                try:
                    q = json.loads(line)
                    task, trial, seed_part = q["episode_id"].rsplit(":", 2)
                    trial = int(trial)
                    seed = int(seed_part.removeprefix("seed-"))
                    qby.setdefault((task, trial, seed), []).append(q)
                except Exception:
                    continue
        rows = []
        for (task, trial), r in by.items():
            if trial >= 30 and mode != "e_vlm_h":
                continue
            key = (task, trial, r["seed"])
            qs = qby.get(key, [])
            if not qs:
                continue
            qs = sorted(qs, key=lambda q: q.get("query_index", -1))
            # target: this skill fails while the paired pi05 succeeds.
            br = baseline_by.get((task, trial))
            if br is None or br["seed"] != r["seed"]:
                continue
            vals = {
                "q_count": len(qs), "steps": r.get("steps", np.nan),
                "k_top1_mean": np.mean([q["units"][0]["score"] for q in qs if q.get("units")]),
                "k_top1_min": np.min([q["units"][0]["score"] for q in qs if q.get("units")]),
                "k_top1_q90": np.quantile([q["units"][0]["score"] for q in qs if q.get("units")], .9),
                "k_gap_mean": np.mean([q.get("margin", np.nan) for q in qs]),
                "k_gap_min": np.nanmin([q.get("margin", np.nan) for q in qs]),
                "k_gap_q90": np.nanquantile([q.get("margin", np.nan) for q in qs], .9),
                "dist_mean": np.mean([q.get("dist_pi05", np.nan) for q in qs]),
                "dist_q90": np.quantile([q.get("dist_pi05", np.nan) for q in qs], .9),
                "t0_mean": np.mean([q.get("t0", np.nan) for q in qs]),
                "t0_q90": np.quantile([q.get("t0", np.nan) for q in qs], .9),
                "history_valid_mean": np.mean([q.get("history_valid", 0) for q in qs]),
                "seconds_query_mean": np.mean([q.get("seconds", np.nan) for q in qs]),
                "unit_repeat_rate": np.mean([int(qs[i]["units"][0]["unit"] == qs[i-1]["units"][0]["unit"])
                                               for i in range(1, len(qs)) if qs[i].get("units") and qs[i-1].get("units")]),
                "skill_success": bool(r["success"]), "pi05_success": bool(br["success"]),
                "target_skill_fail_pi05_success": bool((not r["success"]) and br["success"]),
            }
            rows.append({"task": task, "trial": trial, "seed": r["seed"], **{k: float(v) if isinstance(v, (np.floating,)) else v for k, v in vals.items()}})
        def summarize(subrows):
            aucs = {}
            for name in ("q_count", "steps", "k_top1_mean", "k_top1_min", "k_top1_q90", "k_gap_mean", "k_gap_min", "k_gap_q90", "dist_mean", "dist_q90", "t0_mean", "t0_q90", "history_valid_mean", "seconds_query_mean", "unit_repeat_rate"):
                raw_auc = auc([x["target_skill_fail_pi05_success"] for x in subrows], [x[name] for x in subrows])
                aucs[name] = {"raw": raw_auc, "risk_oriented": (1 - raw_auc if name in ("k_top1_mean", "k_top1_min", "k_gap_mean") and np.isfinite(raw_auc) else raw_auc),
                              "risk_direction": ("lower_is_risk" if name in ("k_top1_mean", "k_top1_min", "k_gap_mean") else "higher_is_risk; exploratory")}
            bytask = {}
            for taskname in ("TurnOnSinkFaucet", "CloseFridge"):
                rr = [x for x in subrows if x["task"] == taskname]
                bytask[taskname] = {k: float(np.nanmean([x[k] for x in rr])) if rr else None for k in ("k_top1_mean", "k_gap_mean", "dist_mean", "t0_mean", "unit_repeat_rate", "steps")}
                bytask[taskname]["n"] = len(rr)
            target = np.asarray([x["target_skill_fail_pi05_success"] for x in subrows], bool)
            trial_steps = np.asarray([x["steps"] for x in subrows], float)
            stagnation = []
            for percentile in (75, 90, 95):
                threshold = float(np.quantile(baseline_success_steps, percentile / 100))
                alarm = trial_steps >= threshold
                stagnation.append({"baseline_success_steps_percentile": percentile, "threshold_steps": threshold,
                                   "target_recall": float((alarm & target).sum() / max(target.sum(), 1)),
                                   "target_precision": float((alarm & target).sum() / max(alarm.sum(), 1)),
                                   "alarm_rate": float(alarm.mean()),
                                   "counterfactual_note": "logged terminal steps only; cannot infer whether switching mid-run changes the outcome"})
            return {"n_trials_with_queries": len(subrows), "target_skill_fail_pi05_success": int(target.sum()), "auc_skill_fail_pi05_success": aucs,
                    "task_comparison": bytask, "stagnation_trigger_from_terminal_steps": stagnation}
        if mode == "e_vlm_h":
            out[mode] = {"trial_0_29": summarize([x for x in rows if x["trial"] < 30]),
                         "trial_0_49": summarize(rows), "trial_features": rows}
        else:
            out[mode] = {**summarize(rows), "trial_features": rows}
    return out


def _closed_condition(mode, limit, baseline_by):
    dat = load_json(ROOT / f"closed/{mode}/eval/results.json")
    rows = [r for r in dat["records"] if r.get("status") == "complete" and r["trial"] < limit]
    lookup = {(r["task"], r["trial"]): r for r in rows}
    common = []
    for key, r in lookup.items():
        b = baseline_by.get(key)
        if b and b["seed"] == r["seed"]:
            common.append((r, b))
    wins = sum(int(r["success"]) for r, _ in common)
    pi = sum(int(b["success"]) for _, b in common)
    sw = sum(r["success"] and not b["success"] for r, b in common)
    sl = sum((not r["success"]) and b["success"] for r, b in common)
    p = float(binomtest(min(sw, sl), sw + sl, .5, alternative="two-sided").pvalue) if sw + sl else 1.0
    tasks = {}
    for name in dat["tasks"]:
        rr = [(r, b) for r, b in common if r["task"] == name]
        tasks[name] = {"n": len(rr), "success": sum(int(r["success"]) for r, _ in rr),
                       "pi05_success": sum(int(b["success"]) for _, b in rr)}
    return {"mode": mode, "requested_trials_per_task": limit, "n_paired": len(common), "success": wins,
            "pi05_success_paired": pi, "discordant_skill_only": int(sw), "discordant_pi05_only": int(sl),
            "mcnemar_exact_two_sided_p": p,
            "mean_steps": float(np.mean([r["steps"] for r, _ in common])) if common else None,
            "median_steps": float(np.median([r["steps"] for r, _ in common])) if common else None,
            "tasks": tasks, "seed_match": len(common) == len(rows)}


def finalize_closed_report():
    status = (ROOT / "closed/lanes.status").read_text()
    if "lane=B EXIT" not in status:
        raise RuntimeError("Refusing final report while lane B remains active")
    base = load_json("/work/.skillnet-state/outputs/robocasa-memory/baseline-01/v6-nosubtask/eval/results.json")
    baseline_by = {(r["task"], r["trial"]): r for r in base["records"] if r.get("status") == "complete"}
    cond = [_closed_condition(m, 30, baseline_by) for m in ("p0", "p5", "e_vlm_h")]
    cond.append(_closed_condition("e_vlm_h", 50, baseline_by))
    common_keys = {(r["task"], r["trial"]) for r in load_json(ROOT / "closed/p5/eval/results.json")["records"] if r.get("status") == "complete" and r["trial"] < 30}
    p5 = {(r["task"], r["trial"]): bool(r["success"]) for r in load_json(ROOT / "closed/p5/eval/results.json")["records"] if r.get("status") == "complete"}
    ev = {(r["task"], r["trial"]): bool(r["success"]) for r in load_json(ROOT / "closed/e_vlm_h/eval/results.json")["records"] if r.get("status") == "complete"}
    union = sum(p5.get(k, False) or ev.get(k, False) for k in common_keys)
    result = {"paired_conditions_vs_pi05": cond, "p5_or_e_vlm_h_union_trials0_29": {"n": len(common_keys), "success": int(union), "provided_upper_bound_reference": {"p5": 117, "e_vlm_h": 114}},
              "lane_switch": "2026-11-17 around 06:50 JST (21:50 UTC Nov 16) to two servers / two lanes; per-lane 8 simulator workers; lane A was stopped after trial 29; lane B completed P0 through trial 29, then exited rc=143 as requested; P0 trials 30-49 were not needed.",
              "baseline": {"success": sum(r["success"] for r in baseline_by.values()), "records": len(baseline_by), "seed_rule": base.get("seed_rule"), "seed": base.get("seed"), "replan_steps": base.get("replan_steps")}}
    (ROOT / "closed/rcx2-final-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# RCX2 閉ループ結果", "", "同一seedで baseline-01/v6-nosubtask と対にした。McNemar は両側 exact binomial。試行seedは task / trial ごとに照合した。", "", "| 条件 | 対象 | 成功 | π0.5同試行 | 条件のみ成功 | π0.5のみ成功 | McNemar p | 平均/中央値 steps |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in cond:
        lines.append(f"| {x['mode']} | 0–{x['requested_trials_per_task']-1}, n={x['n_paired']} | {x['success']}/{x['n_paired']} | {x['pi05_success_paired']}/{x['n_paired']} | {x['discordant_skill_only']} | {x['discordant_pi05_only']} | {x['mcnemar_exact_two_sided_p']:.4f} | {x['mean_steps']:.1f}/{x['median_steps']:.1f} |")
    lines += ["", "## タスク別（trial 0–29）", "", "| 条件 | " + " | ".join(load_json(ROOT / "closed/p5/eval/results.json")["tasks"]) + " |", "|---|" + "---:|" * 6]
    for x in cond[:3]:
        lines.append("| " + x["mode"] + " | " + " | ".join(f"{x['tasks'][t]['success']}/{x['tasks'][t]['n']} (π0.5 {x['tasks'][t]['pi05_success']})" for t in x["tasks"]) + " |")
    lines += ["", f"P5またはE-VLM+Hの試行ごとの和集合は trial 0–29 で {union}/{len(common_keys)}。依頼に記された参照（P5=117、E-VLM+H=114）は固定構成の成功数ではなく、現行の固定構成ログからは同じ数値を再現できなかったため、差を保持して記録する。", "06:50 JST 前後に1サーバーから2サーバー/2レーンへ切り替えた。各レーンのシミュレーター数は8。lane A は不要な30–49を実施せず停止し、lane B はP0のtrial 0–29を完了後、不要な30–49を始める前にrc=143で終了した。", "", "各モードの問い合わせ別ログは `closed/{p0,p5,e_vlm_h}/queries.jsonl`、試行結果は対応する `eval/results.json`。"]
    (ROOT / "closed/report.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--finalize-closed", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.finalize_closed:
        result = finalize_closed_report()
        log_audit = closed_log_audit()
        p = OUT / "signals.json"
        existing = json.loads(p.read_text()) if p.exists() else {"dev": dev_audit()}
        existing["closed_log_audit"] = log_audit
        existing["closed_report_summary"] = result
        p.write_text(json.dumps(existing, indent=2, allow_nan=True) + "\n")
        print(f"wrote {ROOT / 'closed/report.md'} and {p}")
        return
    out = {"dev": dev_audit()}
    if not args.offline_only:
        out["closed_log_audit"] = closed_log_audit()
    (OUT / "signals.json").write_text(json.dumps(out, indent=2, allow_nan=True) + "\n")
    print(f"wrote {OUT / 'signals.json'}")


if __name__ == "__main__":
    main()
