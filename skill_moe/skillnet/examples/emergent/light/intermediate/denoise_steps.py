"""Sweep denoising step counts for normal F, retrieved RAGDP, and (iii)."""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .abstracted_m1 import (
    _decode_residual_draws, _fit_abstracted_regressor, _prepare_m1_train,
)
from .clip_retrieval import (
    CLIP_EPOCHS, CLIP_PATIENCE, _query_embedding,
    fit_dual_encoder, retrieve_embedding_top1,
)
from .splits import SPLITS, subset_rows as _subset
from .experiment import (
    DEFAULT_CACHE, DEFAULT_DATASET, DEFAULT_OUTPUT_ROOT, HORIZONS,
    _standardize, integrate_flow_sources, load_rows, train_flow,
)

STEP_COUNTS = (1, 2, 3, 5, 10)
T0_VALUES = (0.4, 0.6, 0.8)
EVAL_DRAWS = 4
BENCH_LARGE_BATCH = 128


def _known_class_split(rows, train_tasks, eval_tasks, horizon):
    train = _subset(rows, np.isin(rows["task_index"], train_tasks))
    train = _subset(train, train[f"valid_h{horizon}"])
    test_full = _subset(rows, np.isin(rows["task_index"], eval_tasks))
    test_full = _subset(test_full, test_full[f"valid_h{horizon}"])
    known = np.isin(test_full["skill_class"], np.unique(train["skill_class"]))
    test = _subset(test_full, known)
    if not len(train["a"]) or not len(test["a"]):
        raise ValueError(f"empty known-class split; H={horizon}, tasks={train_tasks}->{eval_tasks}")
    if np.intersect1d(train["task_index"], test["task_index"]).size:
        raise AssertionError("held-out suite tasks overlap training tasks")
    return train, test, int(len(test_full["a"]) - len(test["a"]))


def _flow_euler(model, norm, rows, noises, *, source=None, start_t=1.0,
                steps=10, device, chunk=192):
    """Integrate a conditional flow from start_t to zero using K Euler steps."""
    n = len(rows["a"])
    action_shape = tuple(norm["action_shape"])
    dim = int(np.prod(action_shape))
    noises = np.asarray(noises, np.float32).reshape(EVAL_DRAWS, n, dim)
    mean, scale = norm["a_mean"], norm["a_scale"]
    if source is not None:
        source_n = _standardize(np.asarray(source).reshape(n, dim), mean, scale)
    else:
        source_n = None
    errors = np.zeros((EVAL_DRAWS, n), np.float64)
    target = np.asarray(rows["a"], np.float32)
    f_mean, f_scale = norm["f_mean"], norm["f_scale"]
    s_mean, s_scale = norm["s_mean"], norm["s_scale"]
    dt = -float(start_t) / steps
    model.eval()
    with torch.no_grad():
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            count = hi - lo
            eps = torch.as_tensor(noises[:, lo:hi].reshape(EVAL_DRAWS * count, dim),
                                  dtype=torch.float32, device=device)
            if source_n is None:
                x = eps.clone()
            else:
                value = torch.as_tensor(source_n[lo:hi], dtype=torch.float32, device=device)
                x = float(start_t) * eps + (1.0 - float(start_t)) * value.repeat(EVAL_DRAWS, 1)
            f = torch.as_tensor(_standardize(rows["f"][lo:hi], f_mean, f_scale),
                                dtype=torch.float32, device=device).repeat(EVAL_DRAWS, 1)
            s = torch.as_tensor(_standardize(rows["s"][lo:hi], s_mean, s_scale),
                                dtype=torch.float32, device=device).repeat(EVAL_DRAWS, 1)
            for step in range(steps):
                t_value = float(start_t) * (1.0 - step / steps)
                t = torch.full((len(x),), t_value, dtype=torch.float32, device=device)
                x = x + dt * model(x, t, f, s)
            decoded = x.reshape(EVAL_DRAWS, count, dim).cpu().numpy() * scale + mean
            truth = target[lo:hi].reshape(count, dim)
            errors[:, lo:hi] = np.mean((decoded - truth[None]) ** 2, axis=-1)
    return float(errors.mean())


def _iii_euler(model, m1, norm, rows, bases, noises, *, t0, steps,
               residual_seed, device, chunk=96):
    """Decode b to a noisy intermediate, then Euler-denoise with xhat0=Fiii."""
    n = len(rows["a"])
    action_shape = tuple(m1.action_shape)
    dim = int(np.prod(action_shape))
    noises = np.asarray(noises, np.float32).reshape(EVAL_DRAWS, n, dim)
    decoded = _decode_residual_draws(m1, bases, rows["f"], rows["s"],
                                     draws=EVAL_DRAWS, seed=residual_seed)
    mids = _standardize(decoded.reshape(EVAL_DRAWS, n, dim), m1.am, m1.ass)
    truth = np.asarray(rows["a"], np.float32).reshape(n, dim)
    errors = np.zeros((EVAL_DRAWS, n), np.float64)
    dt = -float(t0) / steps
    model.eval()
    with torch.no_grad():
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            count = hi - lo
            # Pair each residual draw with each shared epsilon draw.
            mixed = ((1.0 - t0) * mids[:, None, lo:hi, :]
                     + t0 * noises[None, :, lo:hi, :])
            x = torch.as_tensor(mixed.reshape(EVAL_DRAWS * EVAL_DRAWS * count, dim),
                                dtype=torch.float32, device=device)
            f_np = _standardize(rows["f"][lo:hi], m1.fm, m1.fs)
            s_np = _standardize(rows["s"][lo:hi], m1.sm, m1.ss)
            f = torch.as_tensor(f_np, dtype=torch.float32, device=device).repeat(EVAL_DRAWS * EVAL_DRAWS, 1)
            s = torch.as_tensor(s_np, dtype=torch.float32, device=device).repeat(EVAL_DRAWS * EVAL_DRAWS, 1)
            for step in range(steps):
                t_value = float(t0) * (1.0 - step / steps)
                prediction_x0 = model(x, f, s)
                # Flow path x_t=t*eps+(1-t)*A: velocity=(x_t-A_hat)/t.
                x = x + dt * (x - prediction_x0) / t_value
            decoded_x = (x.reshape(EVAL_DRAWS, EVAL_DRAWS, count, dim).cpu().numpy()
                        * m1.ass + m1.am)
            truth_chunk = truth[lo:hi]
            # Average residual/noise combinations into one error per query.
            err = np.mean((decoded_x - truth_chunk[None, None]) ** 2, axis=-1)
            errors[:, lo:hi] = err.mean(axis=1)
    return float(errors.mean())


def _time_cuda(fn, *, device, rows, repeats):
    if device.type != "cuda":
        raise ValueError("GPU timing requires a CUDA device")
    for _ in range(2):
        fn()
    torch.cuda.synchronize(device)
    timings = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000.0 / rows)
    return float(np.median(timings))


def _benchmark_configs(flow, flow_norm, m1, fiii, query_rows, retrieved_b,
                       retrieved_a, *, device, seed):
    output = {}
    generator = np.random.default_rng(seed)
    for method in ("normal_F", "RAGDP", "iii"):
        for t0 in ((None,) if method == "normal_F" else T0_VALUES):
            for steps in STEP_COUNTS:
                for batch in (1, BENCH_LARGE_BATCH):
                    row = _subset(query_rows, np.arange(len(query_rows["a"])) < batch)
                    count = len(row["a"])
                    noises = generator.standard_normal(
                        (EVAL_DRAWS, count, int(np.prod(row["a"].shape[1:])))).astype(np.float32)
                    if method == "normal_F":
                        fn = lambda row=row, noises=noises, steps=steps: _flow_euler(
                            flow, flow_norm, row, noises, steps=steps, device=device)
                    elif method == "RAGDP":
                        action_source = retrieved_a[:count]
                        fn = lambda row=row, noises=noises, steps=steps, t0=t0, src=action_source: _flow_euler(
                            flow, flow_norm, row, noises, source=src, start_t=t0,
                            steps=steps, device=device)
                    else:
                        base_source = retrieved_b[:count]
                        # Include the per-interval decode in (iii)'s timing.
                        def fn(row=row, noises=noises, steps=steps, t0=t0,
                               bases=base_source, seed_i=seed + steps):
                            return _iii_euler(fiii, m1, flow_norm, row, bases, noises,
                                              t0=t0, steps=steps,
                                              residual_seed=seed_i, device=device)
                    ms = _time_cuda(fn, device=device, rows=count,
                                    repeats=7 if batch == 1 else 4)
                    output[f"{method}|t0={t0}|K={steps}|batch={batch}"] = ms
    return output


def _report(summary):
    lines = [
        "# Denoising step-count sweep",
        "",
        "Retrieval is asymmetric CLIP top-1 without a subtask input. Split A evaluates known-class goal rows; Split B evaluates spatial rows. Each MSE averages four shared epsilon draws. RAGDP starts from the retrieved raw action; (iii) decodes retrieved b with four residual draws and uses Fiii's x0 estimate to form the Euler velocity. `iii direct` is a single Fiii prediction from the same t0-corrupted decode and is identical to the K=1 Euler endpoint. `gold b` uses the query's encoded b as the (iii) upper bound.",
        "",
        "Timing is synchronized CUDA wall time per interval; (iii) includes decoding b plus denoising. Large batch is 128. Retrieval and host-side query embedding are excluded.",
        "",
        "| Split | H | t0 | K | Normal F MSE | RAGDP MSE | RAGDP min K | (iii) MSE | Gold b (iii) MSE | (iii) min K | Direct Fiii K=1 | Normal ms B1/B128 | RAGDP ms B1/B128 | (iii) ms B1/B128 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        fmt = lambda x: "—" if x is None else f"{x:.4f}"
        lines.append(
            f"| {row['split']} | {row['horizon']} | {row['t0']:.1f} | {row['steps']} | {fmt(row['normal_F_mse'])} | {fmt(row['RAGDP_mse'])} | {row['RAGDP_min_K']} | {fmt(row['iii_mse'])} | {fmt(row['iii_gold_b_mse'])} | {row['iii_min_K']} | {fmt(row['iii_direct_regression_mse'])} | {row['normal_ms_per_interval']} | {row['RAGDP_ms_per_interval']} | {row['iii_ms_per_interval']} |"
        )
    lines.extend(["", "`min K` is the smallest tested K whose MSE is no greater than the normal F K=10 MSE; `>10` means no tested K reached it.", ""])
    return "\n".join(lines)


def run(*, output_root=DEFAULT_OUTPUT_ROOT, cache_dir=DEFAULT_CACHE,
        dataset_dir=DEFAULT_DATASET, seed=0, device_name="cuda",
        flow_epochs=150):
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device(device_name)
    rows = load_rows(cache_dir, dataset_dir, max_horizon=max(HORIZONS))
    all_rows = []
    model_info = []
    for split_name, (train_tasks, eval_tasks, _) in SPLITS.items():
        for horizon in HORIZONS:
            train, test, excluded = _known_class_split(
                rows, train_tasks, eval_tasks, horizon)
            train_actions = train["a"][:, :horizon].copy()
            test["a"] = test["a"][:, :horizon].copy()
            flow, flow_norm, flow_fit = train_flow(
                train, horizon=horizon, epochs=flow_epochs, batch_size=512,
                seed=seed + horizon, device=device, patience=8)
            m1 = _prepare_m1_train(train, train_actions, seed=seed + horizon,
                                   epochs=60, batch_size=512, device=device)
            train_b, _ = m1.encode(train_actions, train["s"])
            test_b, _ = m1.encode(test["a"], test["s"])
            train_mids = _decode_residual_draws(
                m1, train_b, train["f"], train["s"], draws=4,
                seed=seed + horizon * 1000 + 11)
            fiii, fiii_fit = _fit_abstracted_regressor(
                m1, train, train_actions, train_b, train_mids,
                epochs=150, batch_size=512, patience=8,
                seed=seed + horizon * 100 + 3, device=device)
            # Fit the no-subtask asymmetric CLIP top-1 and keep row indices so
            # both the retrieved base and raw action use the identical memory item.
            clip_seed = seed + horizon * 17
            torch.manual_seed(clip_seed)
            clip_model, clip_fit_again = fit_dual_encoder(
                train, train_b, use_subtask=False, subtask_mode="none",
                nearby_positive=False, seed=clip_seed,
                epochs=CLIP_EPOCHS, patience=CLIP_PATIENCE, device=device,
                allowed_task_indices=tuple(sorted(np.unique(train["task_index"]).tolist())))
            q = _query_embedding(clip_model, clip_fit_again, test, device=device)
            h = clip_model.encode_skill(torch.as_tensor(train_b, dtype=torch.float32,
                                                         device=device)).detach().cpu().numpy()
            retrieved_indices = retrieve_embedding_top1(q, h)
            retrieved_b = train_b[retrieved_indices]
            retrieved_a = train_actions[retrieved_indices]

            eval_seed = seed + horizon * 3301
            noise_rng = np.random.default_rng(eval_seed)
            noises = [noise_rng.standard_normal((len(test["a"]), int(np.prod(test["a"].shape[1:])))).astype(np.float32)
                      for _ in range(EVAL_DRAWS)]
            noise_stack = np.stack(noises)
            # Normal F K=10 is the shared target for minimum-K calculations.
            normal_by_k = {}
            for steps in STEP_COUNTS:
                normal_by_k[steps] = _flow_euler(
                    flow, flow_norm, test, noise_stack, steps=steps, device=device)
            normal_reference = normal_by_k[10]
            # Match the previous (iii) inference seed convention.
            iii_residual_seed = eval_seed + 100
            retrieved_values, gold_values, direct_values = {}, {}, {}
            for t0 in T0_VALUES:
                retrieved_values[t0] = {}
                gold_values[t0] = {}
                direct_values[t0] = None
                for steps in STEP_COUNTS:
                    retrieved_values[t0][steps] = _iii_euler(
                        fiii, m1, flow_norm, test, retrieved_b, noise_stack,
                        t0=t0, steps=steps, residual_seed=iii_residual_seed,
                        device=device)
                    gold_values[t0][steps] = _iii_euler(
                        fiii, m1, flow_norm, test, test_b, noise_stack,
                        t0=t0, steps=steps, residual_seed=iii_residual_seed,
                        device=device)
                    if steps == 1:
                        direct_values[t0] = retrieved_values[t0][steps]
            ragdp_values = {t0: {steps: _flow_euler(
                flow, flow_norm, test, noise_stack, source=retrieved_a,
                start_t=t0, steps=steps, device=device)
                for steps in STEP_COUNTS} for t0 in T0_VALUES}
            best_k = {}
            for t0 in T0_VALUES:
                for method, values in (("RAGDP", ragdp_values[t0]),
                                       ("iii", retrieved_values[t0])):
                    reached = [k for k in STEP_COUNTS if values[k] <= normal_reference + 1e-12]
                    best_k[(method, t0)] = min(reached) if reached else ">10"

            timings = _benchmark_configs(
                flow, flow_norm, m1, fiii, test, retrieved_b, retrieved_a,
                device=device, seed=seed + horizon * 9001)
            for t0 in T0_VALUES:
                for steps in STEP_COUNTS:
                    all_rows.append({
                        "split": split_name, "horizon": horizon, "t0": t0,
                        "steps": steps, "normal_F_mse": normal_by_k[steps],
                        "RAGDP_mse": ragdp_values[t0][steps],
                        "RAGDP_min_K": best_k[("RAGDP", t0)],
                        "iii_mse": retrieved_values[t0][steps],
                        "iii_gold_b_mse": gold_values[t0][steps],
                        "iii_min_K": best_k[("iii", t0)],
                        "iii_direct_regression_mse": direct_values[t0] if steps == 1 else None,
                        "normal_ms_per_interval": f"{timings[f'normal_F|t0=None|K={steps}|batch=1']:.3f}/"
                                                  f"{timings[f'normal_F|t0=None|K={steps}|batch={BENCH_LARGE_BATCH}']:.3f}",
                        "RAGDP_ms_per_interval": f"{timings[f'RAGDP|t0={t0}|K={steps}|batch=1']:.3f}/"
                                                 f"{timings[f'RAGDP|t0={t0}|K={steps}|batch={BENCH_LARGE_BATCH}']:.3f}",
                        "iii_ms_per_interval": f"{timings[f'iii|t0={t0}|K={steps}|batch=1']:.3f}/"
                                               f"{timings[f'iii|t0={t0}|K={steps}|batch={BENCH_LARGE_BATCH}']:.3f}",
                    })
            model_info.append({
                "split": split_name, "horizon": horizon,
                "training_rows": len(train_actions), "query_rows": len(test["a"]),
                "excluded_unknown_class_rows": excluded,
                "flow_best_epoch": flow_fit["best_epoch"],
                "flow_best_validation_loss": flow_fit["best_validation_loss"],
                "m1_epochs": m1.epochs_run,
                "fiii_best_epoch": fiii_fit["best_epoch"],
                "fiii_best_validation_loss": fiii_fit["best_validation_loss"],
                "clip_best_epoch": clip_fit_again["best_epoch"],
                "clip_best_validation_loss": clip_fit_again["best_validation_loss"],
                "normal_F_10step_mse": normal_reference,
                "timing_batch_sizes": [1, BENCH_LARGE_BATCH],
            })
            print(f"Done split={split_name} H={horizon}; normal F K10={normal_reference:.5f}; "
                  f"queries={len(test['a'])}", flush=True)

    summary = {"seed": seed, "cache": str(cache_dir), "dataset": str(dataset_dir),
               "retrieval": "CLIP top-1, no subtask, asymmetric g(query) dot h(b)",
               "splits": {key: {"train_tasks": list(value[0]), "eval_tasks": list(value[1])}
                          for key, value in SPLITS.items()},
               "step_counts": list(STEP_COUNTS), "t0_values": list(T0_VALUES),
               "iii_t0_direct_regression": "Fiii directly predicts clean A from Dec(b,r,s) corrupted at t0; same as K=1 Euler endpoint",
               "iii_euler_rule": "x_{t-dt}=x_t+dt*(x_t-Fiii(x_t,f,s))/t; Fiii output is x0 estimate",
               "iii_residual_draws": EVAL_DRAWS,
               "timing_note": "synchronized CUDA wall time per interval; includes Dec for iii; excludes CLIP query/retrieval; batch sizes 1 and 128",
               "model_fits": model_info, "rows": all_rows}
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / (datetime.now(timezone.utc).strftime("denoise-step-sweep-%Y%m%dT%H%M%SZ")
                            + "-" + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    (output / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (output / "report.md").write_text(_report(summary))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow-epochs", type=int, default=150)
    args = parser.parse_args(argv)
    output = run(output_root=args.output_root, cache_dir=args.cache_dir,
                 dataset_dir=args.dataset, seed=args.seed,
                 device_name=args.device, flow_epochs=args.flow_epochs)
    print(f"output: {output}")


if __name__ == "__main__":
    main()
