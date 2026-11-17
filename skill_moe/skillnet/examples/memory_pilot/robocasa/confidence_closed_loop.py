"""Fit RCX3-B confidence models and launch reproducible two-server rounds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from confidence_signal_audit import _scores_and_features

OUT = Path("/work/.skillnet-state/outputs/robocasa-memory/latent-skill-01")
CONF = OUT / "conf/closed"
FEATURES_7 = ["k_top1", "k_gap", "k_entropy10", "r_spread10", "r_p5_vs_e_top1", "c_draft_jump", "c_same_unit_repeat"]
FEATURE_S = "s_t02_vs_t08_action_rms"


def _rho(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(spearmanr(a, b).statistic)


def _fit(features, target):
    X = np.column_stack([features[k] for k in FEATURES_7]).astype(np.float64)
    center = np.nanmedian(X, axis=0)
    scale = np.nanstd(X, axis=0)
    scale[scale < 1e-8] = 1.0
    Z = np.nan_to_num((X - center) / scale)
    y = np.log(np.maximum(np.asarray(target, np.float64), 1e-6))
    beta = np.linalg.solve(Z.T @ Z + .01 * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    intercept = float(np.mean(y - Z @ beta))
    return {"features": FEATURES_7, "center": center.tolist(), "scale": scale.tolist(),
            "coefficients": beta.tolist(), "intercept": intercept,
            "spearman_log_error_dev": _rho(Z @ beta + intercept, target)}


def _predict(model, feature_dict):
    x = np.asarray([feature_dict[k] for k in model["features"]], np.float64)
    z = (x - np.asarray(model["center"])) / np.asarray(model["scale"])
    z = np.nan_to_num(z)
    return float(np.exp(np.clip(model["intercept"] + z @ np.asarray(model["coefficients"]), -20, 20)))


def _dev_policy_table(mode, model, feature_dict, draft_error, expert_grid, pi05_error, t0_grid):
    pred = np.asarray([_predict(model, {k: np.asarray(v)[i] for k, v in feature_dict.items()})
                       for i in range(len(draft_error))])
    table = {}
    for pct in (0, 20, 40):
        b = float(np.nextafter(pred.max(), np.inf)) if pct == 0 else float(np.quantile(pred, 1 - pct / 100))
        fallback = pred >= b
        best = None
        for aq in (10, 20, 30, 40, 50, 60):
            a = float(np.quantile(pred, aq / 100))
            if not a < b:
                continue
            t0 = np.full(len(pred), .1)
            ramp = (pred > a) & (pred < b)
            t0[ramp] = .1 + .8 * (pred[ramp] - a) / (b - a)
            gi = np.abs(t0[:, None] - t0_grid[None, :]).argmin(1)
            mse = float(np.where(fallback, pi05_error, expert_grid[np.arange(len(pred)), gi]).mean())
            row = {"fallback_target_percent": pct, "fallback_dev_rate": float(fallback.mean()),
                   "a_quantile_percent": aq, "a": a, "b": b, "dev_mse": mse,
                   "mean_t0_nonfallback": float(t0[~fallback].mean()) if (~fallback).any() else None}
            if best is None or row["dev_mse"] < best["dev_mse"]:
                best = row
        if best is None:
            raise RuntimeError(f"no a < b candidate for {mode} {pct}%")
        table[str(pct)] = best
    return table


def prepare():
    raw = json.loads((OUT / "conf/signals.json").read_text())["dev"]
    rows = dict(np.load(OUT / "rows/dev.npz"))
    per = dict(np.load(OUT / "expert/eval/dev/per_row.npz"))
    pi = per["pi05_arm10"].astype(float)
    grid = per["t0_grid"].astype(float)
    eval_rows = dict(np.load(OUT / "rows/eval.npz"))
    eval_per = dict(np.load(OUT / "expert/eval/eval/per_row.npz"))
    eval_units = dict(np.load(OUT / "units/memory_units.npz"))
    eval_first10 = dict(np.load(OUT / "expert/eval/eval/first10.npz"))
    eval_drafts = {m: dict(np.load(OUT / f"expert/drafts/eval_{m}.npz")) for m in ("p5", "e_vlm_h")}
    prepared, comparison = {}, {}
    for mode in ("p5", "e_vlm_h"):
        mode_data = raw["modes"][mode]
        feat = {k: np.asarray(mode_data["features"][k], float) for k in FEATURES_7 + [FEATURE_S]}
        draft = per[f"draft__{mode}__arm10"].astype(float)
        model7 = _fit(feat, draft)
        # The 8-feature RCX3-A fit is reconstructed on the same Dev rows for a fair check.
        x8_names = FEATURES_7 + [FEATURE_S]
        X8 = np.column_stack([feat[k] for k in x8_names]).astype(np.float64)
        c8, s8 = np.nanmedian(X8, axis=0), np.nanstd(X8, axis=0)
        s8[s8 < 1e-8] = 1.0
        Z8 = np.nan_to_num((X8 - c8) / s8)
        y = np.log(np.maximum(draft, 1e-6))
        b8 = np.linalg.solve(Z8.T @ Z8 + .01 * np.eye(Z8.shape[1]), Z8.T @ (y - y.mean()))
        rho8 = _rho(Z8 @ b8 + y.mean(), draft)
        rho7 = model7["spearman_log_error_dev"]
        selected_features = FEATURES_7 if abs(rho7 - rho8) <= .02 else x8_names
        if selected_features == FEATURES_7:
            chosen = model7
            no_s = True
        else:
            chosen = {"features": x8_names, "center": c8.tolist(), "scale": s8.tolist(),
                      "coefficients": b8.tolist(), "intercept": float(y.mean()), "spearman_log_error_dev": rho8}
            no_s = False
        policies = _dev_policy_table(mode, chosen, feat, draft, per[f"expert__{mode}__arm10"].astype(float), pi, grid)

        # Held-out Eval correlation is confirmation only, never used to choose the signal set or thresholds.
        eval_feat = _scores_and_features(mode, eval_units, eval_rows, eval_drafts[mode], "eval")
        low = eval_first10[f"expert__{mode}__0.2"].astype(np.float32)[..., :6]
        high = eval_first10[f"expert__{mode}__0.8"].astype(np.float32)[..., :6]
        eval_feat[FEATURE_S] = np.sqrt(np.mean((low - high) ** 2, axis=(1, 2)))
        ef = {k: np.asarray(eval_feat[k], float) for k in FEATURES_7 + [FEATURE_S]}
        Xev = np.column_stack([ef[k] for k in chosen["features"]])
        Zev = np.nan_to_num((Xev - np.asarray(chosen["center"])) / np.asarray(chosen["scale"]))
        logev = Zev @ np.asarray(chosen["coefficients"]) + chosen["intercept"]
        rho_eval = _rho(logev, eval_per[f"draft__{mode}__arm10"])
        prepared[mode] = {"mode": mode, "run": json.loads((OUT / "key/selected_configs.json").read_text())["selected"][mode]["run"],
                          "feature_model": chosen, "uses_self_consistency_S": not no_s,
                          "a_b_policies": policies, "spearman_log_error_eval_confirmation": rho_eval,
                          "dev_rows": int(len(draft)), "eval_rows": int(len(eval_rows["task"]))}
        comparison[mode] = {"rho_7_signal_dev": rho7, "rho_8_signal_dev": rho8,
                            "absolute_difference": abs(rho7 - rho8), "selected": "7-signal" if no_s else "8-signal",
                            "eval_spearman_selected_model": rho_eval}
    CONF.mkdir(parents=True, exist_ok=True)
    (CONF / "models.json").write_text(json.dumps(prepared, indent=2) + "\n")
    (CONF / "signal_selection.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare",))
    a = p.parse_args()
    if a.command == "prepare":
        prepare()


if __name__ == "__main__":
    main()
