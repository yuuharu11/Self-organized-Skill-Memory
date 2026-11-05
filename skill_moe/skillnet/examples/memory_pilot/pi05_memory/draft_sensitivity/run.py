"""Offline test of whether the pi05-memory40 correction expert uses draft content."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent
REPO = HERE.parents[4]
import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
_PI05_ROOT = Path(_os.environ.get('SKILLNET_PI05_ROOT', _STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104'))
RUN = _PI05_ROOT / 'run-01'
OUT = _PI05_ROOT / 'draft-sensitivity-01'
CODEC_KEYS = _PI05_ROOT / 'score-audit-01/keys/codec_prior_b_episode.npz'
OFFICIAL = _STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero'
NOISE_SEEDS = (4101, 4102)
MAIN_T0 = 0.6
BATCH = 4  # calibrate.py uses this chunking for bridge keys.

sys.path.insert(0, str(PI05))


def calibration_episodes(rows):
    dev, evaluation = [], []
    for task in np.unique(rows["task"]):
        episodes = np.unique(rows["episode"][(rows["task"] == task) & rows["validation"]])
        order = np.random.default_rng(20261105 + int(task)).permutation(episodes)
        dev.extend(order[:3].tolist())
        evaluation.extend(order[3:].tolist())
    if set(dev) & set(evaluation):
        raise ValueError("calibration episode leakage")
    return np.asarray(dev), np.asarray(evaluation)


def fixed_shuffle(ids, rows):
    rng = np.random.default_rng(20261104)
    task = np.asarray(rows["task"])[ids]
    source = np.empty(len(ids), np.int64)
    by_task = {int(t): ids[task == t].copy() for t in np.unique(task)}
    for t, group in by_task.items():
        rng.shuffle(group)
        other = np.concatenate([values for other_t, values in by_task.items() if other_t != t])
        rng.shuffle(other)
        source[np.flatnonzero(task == t)] = other[rng.integers(0, len(other), size=len(group))]
    if np.any(np.asarray(rows["task"])[source] == task):
        raise AssertionError("shuffled draft source must always be from another task")
    return source


def load_inputs(run: Path, row_limit: int | None):
    rows = {p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in (run / "rows").glob("*.npy")}
    dev_eps, eval_eps = calibration_episodes(rows)
    ids = np.flatnonzero(rows["validation"] & np.isin(rows["episode"], eval_eps))
    if len(ids) != 1318:
        raise ValueError(f"expected 1,318 calibration Eval rows; got {len(ids)}")
    if row_limit is not None:
        # Deterministic approximately equal task allocation, then retain row order.
        chosen = []
        tasks = np.asarray(rows["task"])[ids]
        per_task = max(1, row_limit // len(np.unique(tasks)))
        for task in np.unique(tasks):
            chosen.extend(ids[tasks == task][:per_task].tolist())
        ids = np.asarray(sorted(chosen), np.int64)
        if len(ids) > row_limit:
            ids = ids[:row_limit]
    return rows, ids, dev_eps, eval_eps


def make_drafts(run: Path, rows, ids):
    from run import restore_codec

    codec = restore_codec(run)
    pooled = np.asarray(np.load(run / "pooled.npy", mmap_mode="r", allow_pickle=False)[ids], np.float32)
    state = np.asarray(rows["state"][ids], np.float32)
    actions = np.asarray(rows["actions"][ids], np.float32)
    predicted_residual = np.asarray(
        np.load(run / "predicted_residual.npy", mmap_mode="r", allow_pickle=False)[ids], np.float32
    )
    b_direct, r_direct = codec.prior(pooled, state)
    direct = codec.decode(b_direct, r_direct, pooled, state)
    b_oracle, _ = codec.encode(actions, state)
    oracle = codec.decode(b_oracle, predicted_residual, pooled, state)

    keydata = np.load(CODEC_KEYS, allow_pickle=False)
    query_to_candidate = dict(zip(keydata["query_rows"].tolist(), keydata["cand_rows"][:, 0].tolist(), strict=True))
    missing = [int(row) for row in ids if int(row) not in query_to_candidate]
    if missing:
        raise ValueError(f"codec prior episode Top-1 keys missing query rows: {missing[:5]}")
    candidate_rows = np.asarray([query_to_candidate[int(row)] for row in ids], np.int64)
    bases = np.load(run / "bases.npy", mmap_mode="r", allow_pickle=False)
    retrieved_b = np.asarray(bases[candidate_rows], np.float32)
    bprior_retrieved = codec.decode(retrieved_b, predicted_residual, pooled, state)
    vlm = np.asarray(np.load(run / "drafts.npy", mmap_mode="r", allow_pickle=False)[ids], np.float32)
    shuffle_source = fixed_shuffle(ids, rows)
    shuffled = np.asarray(np.load(run / "drafts.npy", mmap_mode="r", allow_pickle=False)[shuffle_source], np.float32)
    zero = np.zeros_like(vlm, np.float32)
    drafts = {
        "oracle": np.asarray(oracle, np.float32),
        "bprior_direct": np.asarray(direct, np.float32),
        "vlm_retrieved": vlm,
        "bprior_retrieved": np.asarray(bprior_retrieved, np.float32),
        "shuffled": shuffled,
        "zero": zero,
    }
    for name, draft in drafts.items():
        if draft.shape != actions.shape or not np.isfinite(draft).all():
            raise ValueError(f"bad {name} draft shape/values: {draft.shape}")
    return drafts, candidate_rows, shuffle_source


def _noise(jax, noise_seed: int, row_id: int):
    key = jax.random.fold_in(jax.random.key(noise_seed), int(row_id))
    return np.asarray(jax.random.normal(key, (10, 32)), np.float32)


def generate(run: Path, checkpoint: Path, ids: np.ndarray, rows, drafts,
             *, batch_size=BATCH, progress_path: Path | None = None):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex, module_jit
    from run import load_policy, obs_batch

    # Exactly the same lookup rule as calibrate.py. Refuse latest/fallback selection.
    best = json.loads((run / "best_bridge.json").read_text())
    best_path = Path(best["path"])
    if best_path != run / "bridge/001500" or int(best["steps"]) != 1500:
        raise ValueError(f"best_bridge.json does not select the specified 1,500-step checkpoint: {best}")
    if checkpoint.resolve() != OFFICIAL.resolve():
        raise ValueError(f"normal reference must use official pi05_libero checkpoint: {checkpoint}")

    base, _ = load_policy(OFFICIAL)
    baseline = module_jit(base.sample_actions, static_argnames=("num_steps",))
    probe_ids = ids[:min(16, len(ids))]
    probe_obs = obs_batch(rows, probe_ids)
    probe_noise = jnp.asarray(np.stack([_noise(jax, NOISE_SEEDS[0], i) for i in probe_ids]))
    normal_before = np.asarray(baseline(jax.random.key(0), probe_obs, num_steps=10, noise=probe_noise))

    filt = nnx.All(nnx.Param, PathRegex(r".*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*"))
    with ocp.PyTreeCheckpointer() as checkpointer:
        partial = checkpointer.restore(best_path)
    state = nnx.state(base, filt)
    state.replace_by_pure_dict(jax.tree.map(jnp.asarray, partial))
    nnx.update(base, state)
    normal_after = np.asarray(baseline(jax.random.key(0), probe_obs, num_steps=10, noise=probe_noise))
    fallback_equal = bool(np.array_equal(normal_before, normal_after))
    if not fallback_equal:
        raise AssertionError("normal sample_actions changed after best_bridge action-expert overlay")
    bridge = module_jit(base.sample_bridge, static_argnames=("num_steps",))

    names = ["normal"]
    conditions: dict[str, tuple[str | None, float | None]] = {"normal": (None, None)}
    for name in ("oracle", "bprior_direct", "vlm_retrieved", "bprior_retrieved", "shuffled", "zero"):
        cname = f"{name}_t0_0.6"
        names.append(cname)
        conditions[cname] = (name, 0.6)
    for t0 in (0.4, 0.2):
        for name in ("oracle", "bprior_direct", "shuffled"):
            cname = f"{name}_t0_{t0:.1f}"
            names.append(cname)
            conditions[cname] = (name, t0)

    actions = np.asarray(rows["actions"][ids, :, :7], np.float32)
    predictions = np.empty((len(names), len(NOISE_SEEDS), len(ids), 10, 7), np.float32)
    started = time.monotonic()
    completed_batches = 0
    for ni, noise_seed in enumerate(NOISE_SEEDS):
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            batch_ids = ids[start:end]
            obs = obs_batch(rows, batch_ids)
            eps = jnp.asarray(np.stack([_noise(jax, noise_seed, i) for i in batch_ids]))
            normal = baseline(jax.random.key(0), obs, num_steps=10, noise=eps)
            predictions[0, ni, start:end] = np.asarray(normal)[:, :, :7]
            bridge_key = jax.random.fold_in(jax.random.key(noise_seed), start)
            for ci, cname in enumerate(names[1:], 1):
                draft_name, t0 = conditions[cname]
                pred = bridge(bridge_key, obs, jnp.asarray(drafts[draft_name][start:end]),
                              jnp.full((end - start,), t0), num_steps=10, noise=eps)
                predictions[ci, ni, start:end] = np.asarray(pred)[:, :, :7]
            completed_batches += 1
            if progress_path is not None and (completed_batches % 20 == 0 or end == len(ids)):
                record = {"event": "infer_progress", "noise_seed": int(noise_seed),
                          "rows_done_in_seed": int(end), "rows_total": len(ids),
                          "batches_done": completed_batches,
                          "elapsed_seconds": time.monotonic() - started,
                          "gpu_used_mib": _gpu_mib()}
                with progress_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
    return names, predictions, actions, fallback_equal, time.monotonic() - started


def _gpu_mib():
    import subprocess
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5).splitlines()[0]
        return int(text.strip())
    except Exception:
        return None


def _row_mse(pred, target):
    return np.mean((np.asarray(pred) - np.asarray(target)) ** 2, axis=(-1, -2))


def _episode_bootstrap(values, episodes, *, seed=20261104, reps=5000):
    episodes = np.asarray(episodes)
    unique = np.unique(episodes)
    by_episode = [np.flatnonzero(episodes == e) for e in unique]
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, np.float64)
    for i in range(reps):
        sampled = rng.integers(0, len(unique), size=len(unique))
        ix = np.concatenate([by_episode[j] for j in sampled])
        draws[i] = values[ix].mean()
    return np.quantile(draws, [0.025, 0.975]).tolist()


def summarize(names, predictions, actions, drafts, rows, ids):
    from scipy.stats import spearmanr

    target = actions[None, None]
    errors = _row_mse(predictions, target)  # [condition, noise, row]
    means = errors.mean(1)
    episodes = np.asarray(rows["episode"])[ids]
    task_ids = np.asarray(rows["task"])[ids]
    normal_idx = names.index("normal")
    result = {"rows": int(len(ids)), "tasks": int(len(np.unique(task_ids))),
              "episodes": int(len(np.unique(episodes))), "conditions": {},
              "pairwise_vs_normal": {}}
    normal = means[normal_idx]
    for ci, name in enumerate(names):
        e = means[ci]
        item = {"mean_mse": float(e.mean()), "median_mse": float(np.median(e)),
                "fraction_mse_le_0.01": float(np.mean(e <= .01))}
        if name != "normal":
            draft_name, t0 = name.rsplit("_t0_", 1)
            draft = drafts[draft_name][:, :, :7]
            draft_mse = _row_mse(draft, np.asarray(actions))
            final_mean = predictions[ci].mean(0)
            draft_final_rms = np.sqrt(np.mean((final_mean - draft) ** 2, axis=(1, 2)))
            rho = spearmanr(draft_mse, e).statistic
            q = np.quantile(draft_mse, [0, .25, .5, .75, 1])
            bins = np.minimum(np.searchsorted(q[1:-1], draft_mse, side="right"), 3)
            quartile = []
            for qi in range(4):
                sel = bins == qi
                quartile.append({"n": int(sel.sum()), "draft_mse_median": float(np.median(draft_mse[sel])) if sel.any() else None,
                                 "final_mse_mean": float(e[sel].mean()) if sel.any() else None})
            item.update({"t0": float(t0), "draft_mse_mean": float(draft_mse.mean()),
                         "draft_mse_median": float(np.median(draft_mse)),
                         "spearman_draft_vs_final_mse": float(rho),
                         "final_to_draft_rms_mean": float(draft_final_rms.mean()),
                         "final_to_draft_rms_median": float(np.median(draft_final_rms)),
                         "final_mse_by_draft_error_quartile": quartile})
        result["conditions"][name] = item
        if name != "normal":
            delta = e - normal
            result["pairwise_vs_normal"][name] = {
                "mean_delta_mse": float(delta.mean()),
                "episode_bootstrap_95ci": _episode_bootstrap(delta, episodes),
                "lower_is_better": True,
            }
    return result, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--checkpoint", type=Path, default=OFFICIAL)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-tag", default="")
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".22")

    rows, eval_ids, dev_eps, eval_eps = load_inputs(args.run, args.row_limit)
    if args.smoke:
        # Use 16 rows that are also in calibrate.py's saved paired Eval subset.
        cal = np.load(args.run / "calibration_errors.npz", allow_pickle=False)
        smoke_ids = cal["row_ids"][~cal["is_dev"]][:16]
        if len(smoke_ids) != 16 or not np.isin(smoke_ids, eval_ids).all():
            raise ValueError("calibration smoke ids are not in the fixed Eval row set")
        ids = smoke_ids
        smoke_ref = {int(row): float(err) for row, err in zip(
            cal["row_ids"], cal["errors"][:, 0], strict=True)}
    else:
        ids = eval_ids
        smoke_ref = None
    drafts, retrieved_rows, shuffle_sources = make_drafts(args.run, rows, ids)
    protocol = {
        "source_run": str(args.run), "selected_bridge": json.loads((args.run / "best_bridge.json").read_text()),
        "normal_checkpoint": str(args.checkpoint), "rows": int(len(ids)),
        "row_ids": ids.tolist(), "tasks": int(len(np.unique(rows["task"][ids]))),
        "episodes": int(len(np.unique(rows["episode"][ids]))),
        "eval_episode_count_all": int(len(eval_eps)), "dev_episode_count_all": int(len(dev_eps)),
        "conditions": ["oracle", "bprior_direct", "vlm_retrieved", "bprior_retrieved", "shuffled", "zero", "normal"],
        "t0_primary": .6, "t0_secondary": {"values": [.4, .2], "conditions": ["oracle", "bprior_direct", "shuffled"]},
        "noise_seeds": list(NOISE_SEEDS), "noise_rule": "jax.random.normal(fold_in(key(noise_seed), row_id), [10,32])",
        "bridge_rng_key_rule": "fold_in(key(noise_seed), batch_start); batch size matches calibrate.py (4)",
        "metric": "MSE active7, averaged over 10 steps and two fixed noise seeds",
        "row_limit": args.row_limit, "smoke": bool(args.smoke),
        "max_runtime_seconds": 2400, "xla_preallocate": os.getenv("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "xla_mem_fraction": os.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        "top1_codec_base_rows": retrieved_rows.tolist(), "shuffled_source_rows": shuffle_sources.tolist(),
    }
    smoke_suffix = f"_{args.smoke_tag}" if args.smoke and args.smoke_tag else ""
    proto_path = args.output / (f"smoke16_protocol{smoke_suffix}.json" if args.smoke else "protocol.json")
    if proto_path.exists():
        raise FileExistsError(proto_path)
    proto_path.write_text(json.dumps(protocol, indent=2) + "\n")
    progress = args.output / (f"smoke16_progress{smoke_suffix}.jsonl" if args.smoke else "progress.jsonl")
    started = time.monotonic()
    names, predictions, actions, fallback_equal, inference_seconds = generate(
        args.run, args.checkpoint, ids, rows, drafts, batch_size=args.batch_size,
        progress_path=progress)
    summary, errors = summarize(names, predictions, actions, drafts, rows, ids)
    if args.smoke:
        normal = np.asarray(predictions[names.index("normal")], np.float32)
        per_row_mse = _row_mse(normal, actions[None])
        calibration_match = max(abs(float(per_row_mse[:, i].mean()) -
                                     np.mean([smoke_ref[int(row)] for row in ids]))
                                for i in range(len(NOISE_SEEDS)))
        # Saved calibration rows store the average over its same two noise seeds.
        paired_noise_mean = float(per_row_mse.mean())
        saved_mean = float(np.mean([smoke_ref[int(row)] for row in ids]))
        if abs(paired_noise_mean - saved_mean) > 1e-6:
            raise AssertionError(f"normal pi05 calibration MSE mismatch: {paired_noise_mean} vs {saved_mean}")
        summary["smoke_checks"] = {
            "fallback_identical_before_after_overlay": fallback_equal,
            "normal_action_shape": list(normal.shape),
            "normal_mse_active7": paired_noise_mean,
            "calibrate_saved_normal_mse_active7": saved_mean,
            "normal_mse_abs_difference": abs(paired_noise_mean - saved_mean),
            "calibrate_episode_rows_exact": True,
            "inference_seconds": inference_seconds,
            "elapsed_seconds_total": time.monotonic() - started,
            "full_rows_estimated_minutes": inference_seconds * len(eval_ids) / len(ids) / 60,
            "gpu_peak_mib": _gpu_mib(),
        }
        result_path = args.output / f"smoke16_results{smoke_suffix}.json"
        result_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary["smoke_checks"], indent=2))
        return

    npz_path = args.output / "predictions_and_errors.npz"
    if npz_path.exists():
        raise FileExistsError(npz_path)
    np.savez_compressed(npz_path, row_ids=ids, task_ids=np.asarray(rows["task"])[ids],
                        episode_ids=np.asarray(rows["episode"])[ids], condition_names=np.asarray(names),
                        noise_seeds=np.asarray(NOISE_SEEDS), actions=actions, predictions=predictions,
                        row_mse=errors, drafts=np.stack([drafts[n] for n in
                            ("oracle", "bprior_direct", "vlm_retrieved", "bprior_retrieved", "shuffled", "zero")]),
                        draft_names=np.asarray(["oracle", "bprior_direct", "vlm_retrieved", "bprior_retrieved", "shuffled", "zero"]))
    summary.update({"checkpoint": str(args.checkpoint), "best_bridge_path": str(args.run / "bridge/001500"),
                    "fallback_identical_before_after_overlay": fallback_equal,
                    "inference_seconds": inference_seconds, "elapsed_seconds_total": time.monotonic() - started,
                    "gpu_peak_mib": _gpu_mib(), "predictions_path": str(npz_path)})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"rows": len(ids), "conditions": len(names), "inference_seconds": inference_seconds,
                      "gpu_peak_mib": summary["gpu_peak_mib"], "predictions": str(npz_path)}, indent=2))


if __name__ == "__main__":
    main()
