"""Run intermediate-action comparisons on both held-out-suite splits."""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .abstracted_m1 import _run_horizon
from .splits import SPLITS, subset_rows as _subset
from .experiment import (
    DEFAULT_CACHE, DEFAULT_DATASET, DEFAULT_OUTPUT_ROOT, HORIZONS, T0_VALUES,
    _episode_train_validation_indices, _onehot, _query_features, _query_key,
    _standardize, _standardize_fit, integrate_flow_sources, load_rows,
    low_frequency_intermediate, make_units, retrieve_top1,
    retrieve_top1_same_class, split_split_a, train_flow, train_v_regressor,
    predict_v,
)

QUERY_MODES = ("none", "concat", "same_class")
RETRIEVAL_T0S = (0.8, 0.6, 0.4)
III_T0S = (0.0, 0.2, 0.4)


def _same_class_alternatives(train_classes, query_classes, selected, seed):
    """Choose a different memory interval from the same subtask class."""
    rng = np.random.default_rng(seed)
    train_classes = np.asarray(train_classes)
    selected = np.asarray(selected, dtype=np.int64)
    result = np.full(len(query_classes), -1, np.int64)
    for i, (class_id, selected_id) in enumerate(zip(query_classes, selected)):
        candidates = np.flatnonzero(train_classes == class_id)
        candidates = candidates[candidates != selected_id]
        if len(candidates):
            result[i] = int(rng.choice(candidates))
    return result


def xhat0_from_velocity(x_t0, velocity, t0):
    """Recover the clean action estimate when flow velocity is eps-A."""
    return np.asarray(x_t0) - float(t0) * np.asarray(velocity)


def _predict_xhat0(model, normalizers, rows, actions, *, t0, seed, draws=4,
                   batch_size=768, device):
    """Estimate A at t0 from x_t0 and the learned flow velocity.

    Since the trained target is eps-A, xhat0=x_t0-t0*F(x_t0,t0,f,s).
    A fixed set of four corruption draws is averaged into one value per item.
    """
    actions = np.asarray(actions, np.float32).reshape(len(actions), -1)
    action_mean = np.asarray(normalizers["a_mean"], np.float32)
    action_scale = np.asarray(normalizers["a_scale"], np.float32)
    action_norm = _standardize(actions, action_mean, action_scale)
    feature = torch.as_tensor(_standardize(rows["f"], normalizers["f_mean"],
                                           normalizers["f_scale"]), device=device)
    state = torch.as_tensor(_standardize(rows["s"], normalizers["s_mean"],
                                          normalizers["s_scale"]), device=device)
    rng = np.random.default_rng(seed)
    estimates = np.zeros_like(action_norm, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for draw in range(draws):
            noise = rng.standard_normal(action_norm.shape).astype(np.float32)
            noisy = t0 * noise + (1.0 - t0) * action_norm
            for start in range(0, len(actions), batch_size):
                stop = min(len(actions), start + batch_size)
                x = torch.as_tensor(noisy[start:stop], device=device)
                t = torch.full((stop - start,), float(t0), device=device)
                velocity = model(x, t, feature[start:stop], state[start:stop])
                xhat = xhat0_from_velocity(x.cpu().numpy(), velocity.cpu().numpy(), t0)
                estimates[start:stop] += xhat
    estimate = (estimates / draws).astype(np.float32)
    return (estimate * action_scale + action_mean).reshape(len(actions), *normalizers["action_shape"])


def _baseline_run(train, test, *, split_name, horizon, seed, epochs, batch_size,
                  patience, device):
    train = _subset(train, train[f"valid_h{horizon}"])
    test = _subset(test, test[f"valid_h{horizon}"])
    if not len(train["a"]) or not len(test["a"]):
        raise ValueError(f"empty split {split_name} at H={horizon}")
    test_actions = test["a"][:, :horizon].copy()
    test_h = dict(test)
    test_h["a"] = test_actions
    flow, flow_norm, flow_fit = train_flow(
        train, horizon=horizon, epochs=epochs, batch_size=batch_size,
        seed=seed + horizon, device=device, patience=patience)
    rng = np.random.default_rng(seed + horizon * 29)
    noises = [rng.standard_normal((len(test_actions), horizon * test_actions.shape[-1])).astype(np.float32)
              for _ in range(4)]
    normal, normal_per_row = integrate_flow_sources(
        flow, flow_norm, test_h, noises, {"normal": None}, 1.0,
        device=device, return_per_row=True)
    normal_per_row = normal_per_row["normal"]

    class_ids = np.unique(np.concatenate([train["skill_class"], test["skill_class"]]))
    train_task = _onehot(train["skill_class"], class_ids)
    test_task = _onehot(test["skill_class"], class_ids)
    key_mean, key_scale = _standardize_fit(_query_key(train))
    qtrain = {mode: _query_features(train, train_task, mode, key_mean, key_scale)
              for mode in QUERY_MODES}
    qtest = {mode: _query_features(test, test_task, mode, key_mean, key_scale)
             for mode in QUERY_MODES}
    blank_unit = np.zeros(len(train["a"]), dtype=np.int64)
    matches = {}
    for mode in QUERY_MODES:
        if mode == "same_class":
            _, _, ix, unmatched = retrieve_top1_same_class(
                qtrain[mode], np.zeros((len(train["a"]), 1), np.float32), blank_unit,
                qtest[mode], train["skill_class"], test["skill_class"])
        else:
            _, _, ix = retrieve_top1(qtrain[mode], np.zeros((len(train["a"]), 1), np.float32),
                                     blank_unit, qtest[mode])
            unmatched = np.zeros(len(test_actions), bool)
        matches[mode] = (ix, unmatched)

    train_flat = train["a"][:, :horizon].reshape(len(train["a"]), -1)
    train_std = _standardize(train_flat, flow_norm["a_mean"], flow_norm["a_scale"])
    test_std = _standardize(test_actions.reshape(len(test_actions), -1),
                            flow_norm["a_mean"], flow_norm["a_scale"])
    train_std = train_std.reshape(len(train["a"]), horizon, -1)
    test_std = test_std.reshape(len(test_actions), horizon, -1)
    target_values = {"raw": (train["a"][:, :horizon].copy(), test_actions)}
    dct_fit = low_frequency_intermediate(train_std, train_std, 0.8)
    dct_train, dct_keep = dct_fit[0], dct_fit[1]
    if dct_keep.any():
        dct_test, _, _ = low_frequency_intermediate(test_std, train_std, 0.8)
        target_values["dct_lowpass"] = (
            (dct_train.reshape(len(train["a"]), -1) * flow_norm["a_scale"] + flow_norm["a_mean"])
            .reshape(len(train["a"]), horizon, -1),
            (dct_test.reshape(len(test_actions), -1) * flow_norm["a_scale"] + flow_norm["a_mean"])
            .reshape(len(test_actions), horizon, -1),
        )

    known = np.isin(test["skill_class"], np.unique(train["skill_class"]))
    class_groups = {"all": np.ones(len(test_actions), bool),
                    "known_class": known, "unseen_class": ~known}
    rows_out = []
    fits = {"flow": {"best_epoch": flow_fit["best_epoch"],
                      "best_validation_loss": flow_fit["best_validation_loss"],
                      "final_validation_loss": flow_fit["history"][-1]["validation_loss"],
                      "early_stopped": flow_fit["early_stopped"],
                      "epochs": len(flow_fit["history"])}}
    random_indices = rng.integers(0, len(train["a"]), size=len(test_actions))
    xhat_values = {}
    for t0 in RETRIEVAL_T0S:
        xhat_values[t0] = (
            _predict_xhat0(flow, flow_norm, train, train["a"][:, :horizon],
                           t0=t0, seed=seed + horizon * 8101 + int(t0 * 100), device=device),
            _predict_xhat0(flow, flow_norm, test, test_actions,
                           t0=t0, seed=seed + horizon * 9101 + int(t0 * 100), device=device),
        )
    for representation, (train_v, test_v) in target_values.items():
        unit = make_units(train_v, train["episode_id"], kappa=0.1,
                          min_support=10, min_episodes=3, pending_batch=200,
                          seed=seed + horizon)
        target_mean, target_scale = flow_norm["a_mean"], flow_norm["a_scale"]
        q_models = {}
        for use_subtask in (False, True):
            qmodel, qnorm, qfit = train_v_regressor(
                train, train_v, _onehot(train["skill_class"], np.unique(train["skill_class"])),
                use_subtask=use_subtask, epochs=min(50, epochs), batch_size=batch_size,
                seed=seed + horizon * 100 + int(use_subtask) * 13 + (representation == "dct_lowpass") * 31,
                device=device, target_mean=target_mean, target_scale=target_scale,
                patience=patience)
            qvalue = predict_v(qmodel, qnorm, test,
                               _onehot(test["skill_class"], np.unique(train["skill_class"])),
                               use_subtask=use_subtask, device=device)
            q_models[use_subtask] = qvalue
            fits[f"Q_{representation}_{'subtask' if use_subtask else 'no_subtask'}"] = {
                "best_epoch": qfit["best_epoch"],
                "best_validation_loss": qfit["best_validation_loss"],
                "epochs": len(qfit["history"])}
        for mode in QUERY_MODES:
            indices, unmatched = matches[mode]
            eval_mask = ~unmatched
            if not eval_mask.any():
                continue
            q_value = q_models[mode != "none"]
            sources = {
                "memory_top1": train_v[indices[eval_mask]],
                "Q": q_value[eval_mask],
                "oracle": test_v[eval_mask],
                "random": train_v[random_indices[eval_mask]],
            }
            eval_rows = _subset(test_h, eval_mask)
            eval_noise = [noise[eval_mask] for noise in noises]
            for t0 in RETRIEVAL_T0S:
                _, per_row = integrate_flow_sources(
                    flow, flow_norm, eval_rows, eval_noise,
                    {k: v.reshape(len(v), -1) for k, v in sources.items()}, t0,
                    device=device, return_per_row=True)
                for group, original_group_mask in class_groups.items():
                    mask = original_group_mask[eval_mask]
                    if not mask.any():
                        continue
                    means = {name: float(np.nanmean(values[mask]))
                             for name, values in per_row.items()}
                    means["normal"] = float(np.nanmean(normal_per_row[eval_mask][mask]))
                    result_row = {"split": split_name, "horizon": horizon,
                                     "method": "ii", "representation": representation,
                                     "eval_t0": t0, "query": mode, "class_group": group,
                                     "rows": int(mask.sum()), "unmatched": int(unmatched.sum()),
                                     "mse": means,
                                     "memory_minus_random": means["memory_top1"] - means["random"],
                                     "Q_minus_memory": means["Q"] - means["memory_top1"]}
                    rows_out.append(result_row)
                    if representation == "raw":
                        # RAGDP is the same stored completed-action value as
                        # (ii/raw), explicitly surfaced as the primary baseline.
                        rows_out.append({**result_row, "method": "RAGDP",
                                         "representation": "retrieved raw A; re-noised at t0"})

                alt_all = _same_class_alternatives(
                    train["skill_class"], test["skill_class"][eval_mask], indices[eval_mask],
                    seed + horizon * 5101 + int(t0 * 100) + int(mode == "concat") * 17
                    + int(mode == "same_class") * 29)
                alt_valid = np.flatnonzero(alt_all >= 0)
                if len(alt_valid):
                    # Compare retrieved and shuffled intervals with identical
                    # query rows and identical diffusion noise.
                    alt_mask = np.zeros(len(eval_rows["a"]), dtype=bool)
                    alt_mask[alt_valid] = True
                    alt_rows = _subset(eval_rows, alt_mask)
                    alt_noise = [noise[alt_valid] for noise in eval_noise]
                    selected_ids = indices[eval_mask][alt_valid]
                    shuffle_values = {
                        "retrieved_same_class": train_v[selected_ids].reshape(len(alt_valid), -1),
                        "shuffled_same_class": train_v[alt_all[alt_valid]].reshape(len(alt_valid), -1),
                    }
                    _, shuffle_per_row = integrate_flow_sources(
                        flow, flow_norm, alt_rows, alt_noise, shuffle_values, t0,
                        device=device, return_per_row=True)
                    for group, original_group_mask in class_groups.items():
                        if group == "all":
                            continue
                        group_mask = original_group_mask[eval_mask][alt_valid]
                        if not group_mask.any():
                            continue
                        retrieved_mse = float(np.nanmean(shuffle_per_row["retrieved_same_class"][group_mask]))
                        shuffled_mse = float(np.nanmean(shuffle_per_row["shuffled_same_class"][group_mask]))
                        rows_out.append({"split": split_name, "horizon": horizon,
                                         "method": "ii-shuffle", "representation": representation,
                                         "eval_t0": t0, "query": mode, "class_group": group,
                                         "rows": int(group_mask.sum()), "unmatched": int(unmatched.sum()),
                                         "mse": {"normal": float(np.nanmean(normal_per_row[eval_mask][alt_valid][group_mask])),
                                                 "memory_top1": shuffled_mse,
                                                 "retrieved_reference": retrieved_mse},
                                         "shuffle_minus_retrieved": shuffled_mse - retrieved_mse})

    for mode in QUERY_MODES:
        indices, unmatched = matches[mode]
        eligible = ~unmatched
        eval_ix = np.flatnonzero(eligible)
        for t0 in RETRIEVAL_T0S:
            train_xhat, test_xhat = xhat_values[t0]
            qsource = {"memory_top1": train_xhat[indices[eval_ix]].reshape(len(eval_ix), -1),
                       "oracle": test_xhat[eval_ix].reshape(len(eval_ix), -1),
                       "random": train_xhat[random_indices[eval_ix]].reshape(len(eval_ix), -1)}
            eval_rows = _subset(test_h, eligible)
            eval_noise = [noise[eligible] for noise in noises]
            _, xhat_per_row = integrate_flow_sources(
                flow, flow_norm, eval_rows, eval_noise, qsource, t0,
                device=device, return_per_row=True)
            for group, original_group_mask in class_groups.items():
                mask = original_group_mask[eligible]
                if not mask.any():
                    continue
                means = {name: float(np.nanmean(values[mask]))
                         for name, values in xhat_per_row.items()}
                means["normal"] = float(np.nanmean(normal_per_row[eligible][mask]))
                rows_out.append({"split": split_name, "horizon": horizon,
                                 "method": "iv", "representation": "xhat0(t0)",
                                 "eval_t0": t0, "query": mode, "class_group": group,
                                 "rows": int(mask.sum()), "unmatched": int(unmatched.sum()),
                                 "mse": means,
                                 "memory_minus_random": means["memory_top1"] - means["random"]})
    for group, mask in class_groups.items():
        if mask.any():
            rows_out.append({"split": split_name, "horizon": horizon, "method": "i",
                             "representation": "normal_F", "eval_t0": None,
                             "query": "—", "class_group": group, "rows": int(mask.sum()),
                             "unmatched": 0, "mse": {"normal": float(np.nanmean(normal_per_row[mask]))}})
    return {"train_rows": len(train["a"]), "test_rows": len(test_actions),
            "test_class_counts": {str(c): int(np.sum(test["skill_class"] == c))
                                  for c in np.unique(test["skill_class"])},
            "train_classes": np.unique(train["skill_class"]).tolist(),
            "known_class_rows": int(known.sum()), "unseen_class_rows": int((~known).sum()),
            "flow_normal_mse": normal["normal"], "fits": fits, "rows": rows_out}


def _report(data):
    lines = ["# Intermediate action study: Split A and Split B", "",
             "Split A trains object+spatial and evaluates goal. Goal rows are split by whether eval_skill_class occurs in training. Split B trains object+goal and evaluates spatial; all reported class IDs are checked against the training class set.", "",
             "RAGDP (retrieved completed raw A, re-noised at t0) is listed first. It is the raw-A retrieval path from (ii), made explicit as the principal retrieval-plus-denoising baseline. (ii) and (iii) also include same-subtask interval shuffles. (iv) stores a clean-action estimate xhat0(t0)=x_t0-t0*F(x_t0,t0,f,s), computed for each training interval from four fixed-seed corruptions and averaged. Query modes are (a) none, (b) normalized subtask concatenation, and (c) same-class filtering followed by nearest image+state key. MSE is masked over valid action dimensions.", "",
             "| Split | H | Class subset | Method / value | Query | eval t0 | rows | normal F | retrieved/alternate | random | Q | oracle | Δ vs random | shuffle−retrieved |",
             "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    method_order = {"RAGDP": 0, "i": 1, "ii": 2, "ii-shuffle": 3, "iv": 4}
    for split_result in data["splits"]:
        split_key = split_result["name"]
        for horizon_run in split_result["horizons"]:
            for row in sorted(horizon_run["rows"], key=lambda r: method_order.get(r["method"], 9)):
                mse = row["mse"]
                value_text = "—"
                if "memory_top1" in mse:
                    if row["method"] == "ii-shuffle":
                        value_text = (f"{mse['retrieved_reference']:.4f} / "
                                      f"{mse['memory_top1']:.4f}")
                    else:
                        value_text = f"{mse['memory_top1']:.4f}"
                lines.append("| {} | {} | {} | {} {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    split_key, row["horizon"], row["class_group"], row["method"],
                    row["representation"], row["query"],
                    "—" if row["eval_t0"] is None else f"{row['eval_t0']:.1f}", row["rows"],
                    f"{mse['normal']:.4f}" if "normal" in mse else "—",
                    value_text,
                    f"{mse['random']:.4f}" if "random" in mse else "—",
                    f"{mse['Q']:.4f}" if "Q" in mse else "—",
                    f"{mse['oracle']:.4f}" if "oracle" in mse else "—",
                    f"{row['memory_minus_random']:+.4f}" if "memory_minus_random" in row else "—",
                    f"{row['shuffle_minus_retrieved']:+.4f}" if "shuffle_minus_retrieved" in row else "—"))
            iii = horizon_run["iii"]
            for row in iii["results"]:
                for group in ("all", "known_class", "unseen_class"):
                    if group == "all":
                        mse = row["mse"]
                        count = row["mse_rows"]
                    elif group == "known_class":
                        mse = row["mse_eligible"]
                        count = row["eligible_query_rows"]
                    else:
                        mse = row["mse_new_class"]
                        count = row["goal_rows"] - row["eligible_query_rows"]
                    if not mse or mse.get("memory_top1") is None:
                        continue
                    lines.append("| {} | {} | {} | iii white-noise | {} | {} | {} | — | {} | {} | — | {} | {} | — |".format(
                        split_key, row["horizon"], group, row["retrieval"],
                        f"{row['white_noise_t0']:.1f}", count,
                        f"{mse['memory_top1']:.4f}", f"{mse['random_train_base']:.4f}",
                        f"{mse['oracle_base']:.4f}",
                        f"{row['mse_new_class']['memory_top1'] - row['mse_new_class']['random_train_base']:+.4f}"
                        if group == "unseen_class" and row.get("mse_new_class") and row["mse_new_class"].get("memory_top1") is not None
                        else (f"{mse['memory_top1'] - mse['random_train_base']:+.4f}"),
                        ))
                shuffle = row.get("shuffle_same_class", {})
                if shuffle:
                    for group, stat in shuffle.items():
                        if group == "all":
                            continue
                        lines.append("| {} | {} | {} | iii shuffled interval | {} | {} | {} | — | {} / {} | — | — | — | — | {:+.4f} |".format(
                            split_key, row["horizon"], group, row["retrieval"],
                            f"{row['white_noise_t0']:.1f}", stat["rows"],
                            f"{stat['retrieved_mse']:.4f}", f"{stat['shuffled_mse']:.4f}",
                            stat["shuffled_mse"] - stat["retrieved_mse"]))
    lines += ["", "## Split support and class coverage", "",
              "| Split | H | train rows | eval rows | training eval_skill_class IDs | eval rows with known class | eval rows with unseen class |",
              "|---|---:|---:|---:|---|---:|---:|"]
    for split_result in data["splits"]:
        for h in split_result["horizons"]:
            lines.append(f"| {split_result['name']} | {h['horizon']} | {h['train_rows']} | {h['test_rows']} | "
                         f"{h['train_classes']} | {h['known_class_rows']} | {h['unseen_class_rows']} |")
    lines += ["", "## Notes", "",
              "For c, unseen classes have no matching memory candidate and are omitted from c's memory MSE; the table shows their unavailable rows in the coverage table. Comparisons across class groups are conditional on rows with valid H-step actions. The same-class (c) query uses eval_skill_class only to filter candidates, then uses the image+state key for nearest-neighbor search.", "",
              "The shuffle control replaces each retrieved value with a different training interval of the same eval_skill_class; its row uses the same query observation and same diffusion noise as its retrieved-reference partner. If a class has no training interval, the shuffle is unavailable and is not scored. For RAGDP, the explicit row duplicates the raw-A retrieval computation by design.", "",
              "(iii) uses M1 (HSIC=0.01, beta_r=1) trained on each split's training suites, resamples its residual, and fits the same-sized Fiii regressor. The goal/spatial evaluation suites are never inserted into memory. H=10 DCT target t0=0.8 is omitted when its signal-dominant coefficient mask is empty. Coarse motion-label diagnostics remain in results.json.", ""]
    return "\n".join(lines)


def run(*, output_root=DEFAULT_OUTPUT_ROOT, cache_dir=DEFAULT_CACHE,
        dataset_dir=DEFAULT_DATASET, seed=0, flow_epochs=150, batch_size=512,
        patience=8, m1_epochs=60, iii_epochs=150, device_name="cuda"):
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device(device_name)
    all_rows = load_rows(cache_dir, dataset_dir, max_horizon=max(HORIZONS))
    split_runs = []
    for name, (train_tasks, test_tasks, description) in SPLITS.items():
        train_mask = np.isin(all_rows["task_index"], train_tasks)
        test_mask = np.isin(all_rows["task_index"], test_tasks)
        train_all, test_all = _subset(all_rows, train_mask), _subset(all_rows, test_mask)
        horizons = []
        for horizon in HORIZONS:
            baseline = _baseline_run(train_all, test_all, split_name=name, horizon=horizon,
                                     seed=seed, epochs=flow_epochs, batch_size=batch_size,
                                     patience=patience, device=device)
            train_h = _subset(train_all, train_all[f"valid_h{horizon}"])
            test_h = _subset(test_all, test_all[f"valid_h{horizon}"])
            iii = _run_horizon(train_h, test_h, horizon=horizon, seed=seed,
                               m1_epochs=m1_epochs, action_epochs=iii_epochs,
                               batch_size=batch_size, patience=patience, device=device)
            for row in iii["results"]:
                row["mse_new_class"] = row.get("mse_new_class", {})
            horizons.append({"horizon": horizon, **baseline, "iii": iii})
            print(f"Split {name} H={horizon}: (i)/(ii) rows={baseline['test_rows']}; "
                  f"(iii) M1 epochs={iii['m1']['epochs_run']}; "
                  f"Fiii best epoch={iii['abstracted_action_head']['best_epoch']}", flush=True)
        split_runs.append({"name": name, "description": description,
                           "train_task_indices": list(train_tasks),
                           "eval_task_indices": list(test_tasks), "horizons": horizons})
    out = output_root / (datetime.now(timezone.utc).strftime("intermediate-action-splits-%Y%m%dT%H%M%SZ")
                         + "-" + uuid.uuid4().hex[:6])
    out.mkdir(parents=True, exist_ok=False)
    summary = {"seed": seed, "cache": str(cache_dir), "dataset": str(dataset_dir),
               "splits": split_runs,
               "settings": {"flow_epochs_max": flow_epochs, "batch_size": batch_size,
                            "patience": patience, "m1_epochs_max": m1_epochs,
                            "iii_epochs_max": iii_epochs, "t0_ii": list(RETRIEVAL_T0S),
                            "t0_iii": list(III_T0S), "noise_draws": 4,
                            "goal_or_spatial_added_to_memory": False}}
    (out / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out / "report.md").write_text(_report(summary))
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flow-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--m1-epochs", type=int, default=60)
    parser.add_argument("--iii-epochs", type=int, default=150)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result_dir = run(output_root=args.output_root, cache_dir=args.cache_dir,
                     dataset_dir=args.dataset, seed=args.seed, flow_epochs=args.flow_epochs,
                     batch_size=args.batch_size, patience=args.patience,
                     m1_epochs=args.m1_epochs, iii_epochs=args.iii_epochs,
                     device_name=args.device)
    print(f"output: {result_dir}")
