"""Run the frozen-pi0.5, small bottleneck go/no-go experiment.

Usage is documented in README.md. Features are cached per episode and a partially
completed extraction resumes without replacing an existing cache file.
"""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from skill_moe.skillnet.examples.emergent.eval.gonogo import GoNoGoCriteria
from skill_moe.skillnet.examples.emergent.eval.metrics import MIConfig
from skill_moe.skillnet.examples.emergent.eval.pipeline import run_all
from .extract import dataset_episodes, extract_cache
from .runtime import seed_experiment as _seed
from .model import BottleneckModel, gaussian_kl, standardize

from ...research_paths import REPO_ROOT as ROOT, STATE_DIR
DEFAULT_DATA = STATE_DIR / "data/lerobot/emergent/libero_40_unsegmented"
DEFAULT_CKPT = STATE_DIR / "checkpoints/openpi-assets/checkpoints/pi05_base/params"
DEFAULT_OUTPUTS = STATE_DIR / "outputs/memory-pilot"


def _quat_from_rotvec(v: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    half = theta * 0.5
    scale = np.divide(np.sin(half), theta, out=np.full_like(theta, 0.5), where=theta > 1e-8)
    return np.concatenate([v * scale, np.cos(half)], axis=-1)


def _load_cached(cache_path: Path) -> dict[str, Any]:
    with np.load(cache_path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _collect(cache_manifest: dict[str, Any], horizon: int):
    """Expand sampled observations into full, fixed H action windows and env trajectories."""
    records = []
    dropped_tail = 0
    full_count = 0
    for meta in cache_manifest["episodes"]:
        d = _load_cached(Path(meta["cache"]))
        states, actions, steps = d["env_state"].astype(np.float32), d["action"].astype(np.float32), d["step"].astype(int)
        all_steps = np.asarray(d.get("env_step", np.arange(len(states)))).astype(int)
        step_to_idx = {int(s): i for i, s in enumerate(all_steps)}
        sampled = d["step"].astype(int)
        full = []
        for step in sampled:
            index = step_to_idx.get(int(step), -1)
            valid = index >= 0 and index + horizon < len(states) and index + horizon <= len(actions)
            if not valid:
                dropped_tail += 1
                full.append(None)
                continue
            full_count += 1
            full.append(actions[index:index + horizon])
        records.append({**meta, "cache_arrays": d, "windows": full})
    return records, {"horizon": horizon, "full_windows": full_count, "tail_windows_dropped": dropped_tail}


def _prepare(records, horizon: int):
    train_eps, eval_eps = [], []
    rng = np.random.default_rng(123)
    by_task: dict[str, list] = {}
    for ep in records:
        by_task.setdefault(ep["task"], []).append(ep)
    for task, eps in sorted(by_task.items()):
        perm = rng.permutation(len(eps))
        n_eval = max(2, int(round(len(eps) * 0.2))) if len(eps) >= 10 else max(1, int(round(len(eps) * .2)))
        n_eval = min(n_eval, len(eps) - 1)
        eval_ids = set(perm[:n_eval].tolist())
        for i, ep in enumerate(eps):
            (eval_eps if i in eval_ids else train_eps).append(ep)
    def rows(eps):
        out = []
        for ep in eps:
            d = ep["cache_arrays"]
            for j, act in enumerate(ep["windows"]):
                if act is None:
                    continue
                step = int(d["step"][j])
                out.append({"ep": ep, "j": j, "step": step, "f": d["f"][j].astype(np.float32),
                            "k": d["k"][j].astype(np.float32), "s": d["proprio"][j].astype(np.float32),
                            "a": act.astype(np.float32)})
        return out
    return rows(train_eps), rows(eval_eps), train_eps, eval_eps


def _fit(train, eval_rows, horizon: int, beta: float, seed: int, *, epochs: int, batch_size: int,
         device: torch.device):
    _seed(seed)
    Ftr = np.stack([r["f"] for r in train]); Str = np.stack([r["s"] for r in train]); Atr = np.stack([r["a"] for r in train])
    F, = standardize(Ftr, Ftr)
    S, = standardize(Str, Str)
    A, = standardize(Atr.reshape(len(Atr), -1), Atr.reshape(len(Atr), -1))
    fm, fs = Ftr.mean(0), Ftr.std(0); fs[fs < 1e-6] = 1
    sm, ss = Str.mean(0), Str.std(0); ss[ss < 1e-6] = 1
    am, ass = Atr.reshape(len(Atr), -1).mean(0), Atr.reshape(len(Atr), -1).std(0)
    ass[ass < 1e-6] = 1
    tensors = TensorDataset(torch.from_numpy(A).float(), torch.from_numpy(S).float(), torch.from_numpy(F).float())
    loader = DataLoader(tensors, batch_size=batch_size, shuffle=True, drop_last=False)
    model = BottleneckModel((horizon, Atr.shape[-1]), Str.shape[-1], Ftr.shape[-1], z_dim=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.train()
    history = []
    for epoch in range(epochs):
        losses = []
        for a, s, f in loader:
            a, s, f = a.to(device), s.to(device), f.to(device)
            pred, qm, qlv, pm, plv = model(a, s, f, sample=True)
            recon = (pred - a.reshape_as(pred)).square().mean()
            kl = gaussian_kl(qm, qlv, pm, plv)
            loss = recon + beta * kl
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append((float(recon.detach()), float(kl.detach())))
        history.append({"epoch": epoch + 1, "recon_mse": float(np.mean([v[0] for v in losses])),
                        "kl_nats": float(np.mean([v[1] for v in losses]))})
    model.eval()
    def norm(rows):
        f = (np.stack([r["f"] for r in rows]) - fm) / fs
        s = (np.stack([r["s"] for r in rows]) - sm) / ss
        a = (np.stack([r["a"] for r in rows]).reshape(len(rows), -1) - am) / ass
        return (torch.from_numpy(a).float().to(device), torch.from_numpy(s).float().to(device),
                torch.from_numpy(f).float().to(device))
    with torch.no_grad():
        # Posterior reference and inference-time prior means.
        a, s, f = norm(eval_rows)
        post_m, post_lv = model.encode(a, s)
        prior_m, prior_lv = model.predict_prior(f, s)
        # Same architecture at initialization only, seeded deterministically.
        torch.manual_seed(seed + 9001)
        untrained = BottleneckModel((horizon, Atr.shape[-1]), Str.shape[-1], Ftr.shape[-1], z_dim=32).to(device).eval()
        um, ulv = untrained.predict_prior(f, s)
    output = {"post_mean": post_m.cpu().numpy(), "post_logvar": post_lv.cpu().numpy(),
              "prior_mean": prior_m.cpu().numpy(), "prior_logvar": prior_lv.cpu().numpy(),
              "untrained_prior_mean": um.cpu().numpy(), "untrained_prior_logvar": ulv.cpu().numpy()}
    normalizers = {"f_mean": fm, "f_scale": fs, "s_mean": sm, "s_scale": ss, "a_mean": am, "a_scale": ass}
    return model, output, normalizers, history


def _write_eval_input(path: Path, eval_eps, eval_rows, preds, horizon: int, model_info: dict[str, Any]):
    path.mkdir(parents=True, exist_ok=False)
    epdir = path / "episodes"; epdir.mkdir()
    row_by_ep: dict[str, list[int]] = {}
    for i, row in enumerate(eval_rows):
        row_by_ep.setdefault(row["ep"]["episode_id"], []).append(i)
    manifest_eps = []
    for ep in eval_eps:
        eid = ep["episode_id"]; d = ep["cache_arrays"]
        rows = row_by_ep.get(eid, [])
        # Include all sampled steps; incomplete tails are present but unusable for this H.
        steps = d["step"].astype(np.int32)
        n = len(steps); dim = 32; action_dim = d["action"].shape[-1]
        post = np.full((n, dim), np.nan, np.float32); post_lv = np.full_like(post, np.nan)
        prior = np.zeros((n, dim), np.float32); prior_lv = np.zeros_like(prior)
        untrained = np.zeros_like(prior)
        z_actions = np.zeros((n, horizon, action_dim), np.float32); z_mask = np.zeros((n, horizon), bool)
        env_steps = d.get("env_step", np.arange(len(d["env_state"]))).astype(int)
        interval = np.minimum(horizon, env_steps[-1] + 1 - steps).clip(0)
        executed = interval > 0
        for ri in rows:
            row = eval_rows[ri]; j = row["j"]
            prior[j] = preds["prior_mean"][ri]; prior_lv[j] = preds["prior_logvar"][ri]
            post[j] = preds["post_mean"][ri]; post_lv[j] = preds["post_logvar"][ri]
            untrained[j] = preds["untrained_prior_mean"][ri]
            z_actions[j] = row["a"]; z_mask[j] = True
        state = d["env_state"].astype(np.float32)
        # LIBERO qpos is xyz, axis-angle xyz, gripper qpos left/right.
        pred_chunk = np.zeros((n, horizon, action_dim), np.float32)
        n_executed = np.zeros(n, np.int32)
        full_steps = d.get("env_step", np.arange(len(d["action"]))).astype(int)
        step_to_action = {int(s): i for i, s in enumerate(full_steps)}
        action_stride = 5
        for j, step in enumerate(steps):
            start = step_to_action.get(int(step), -1)
            if start >= 0:
                n_exec = min(action_stride, horizon, len(d["action"]) - start)
                n_executed[j] = n_exec
                pred_chunk[j, :n_exec] = d["action"][start:start + n_exec]
        arrays = {"step": steps, "proprio": d["proprio"].astype(np.float32), "pred_chunk": pred_chunk,
                  "n_executed": n_executed, "z_post_mean": post, "z_post_logvar": post_lv,
                  "z_prior_mean": prior, "z_prior_logvar": prior_lv, "z_untrained_prior_mean": untrained,
                  "z_interval_len": interval.astype(np.int32), "z_executed": executed,
                  # Legacy crosstask API compatibility; pick/place are derived eval proxies,
                  # not source annotations. The actual executed gripper commands above
                  # define their temporal transition.
                  "skills": np.tile(np.array([[1, 2]], dtype=np.int32), (n, 1)),
                  "skill_mask": np.ones((n, 2), dtype=bool),
                  "key__vlm_img_nolang": d["k"].astype(np.float32), "key__vlm_img_lang": d["f"].astype(np.float32),
                  "z_target_actions": z_actions, "z_target_mask": z_mask,
                  "env_eef_pos": state[:, :3], "env_eef_quat": _quat_from_rotvec(state[:, 3:6]),
                  "env_gripper": state[:, 6:8]}
        with (epdir / f"{eid}.npz").open("xb") as f:
            np.savez_compressed(f, **arrays)
        manifest_eps.append({"episode_id": eid, "task": ep["task"], "task_id": ep["task_index"],
                             "suite": ep["suite"], "init_state_id": eid, "success": None,
                             "steps": int(len(state)), "env_step_offset": 0})
    manifest = {"format_version": 1, "suite": "libero_spatial_object_goal", "replan_steps": 5,
                "env_step_offset": 0, "episodes": manifest_eps,
                "emergent": {"schema_version": 1, "z_horizon": horizon, "z_dim": 32,
                             "model": "light-frozen-pi05-prior-decoder", "checkpoint": model_info["checkpoint"],
                             "untrained_checkpoint": f"random init seed {model_info['seed'] + 9001}",
                             "posterior_input": f"executed actions [step, step+{horizon}) + proprio at step",
                             "prior_input": "pi0.5 PaliGemma language-conditioned image tokens + proprio",
                             "key": {"vlm_img_nolang": {"model": "pi0.5 base PaliGemma", "tokens": "base+wrist image",
                                                         "pooling": "mean", "language_input": False},
                                     "vlm_img_lang": {"model": "pi0.5 base PaliGemma", "tokens": "base+wrist image",
                                                      "pooling": "mean", "language_input": True}},
                             "training_episodes": model_info["training_episodes"],
                             "training": {"beta": model_info["beta"], "objective": "normalized action MSE + beta * analytic KL(q||p)",
                                          "optimizer": "AdamW", "epochs": model_info["epochs"], "seed": model_info["seed"]}}}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run(args):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    out_root = args.output_root / f"emergent-light-{run_id}"
    out_root.mkdir(parents=True, exist_ok=False)
    cache = args.cache_dir or (out_root / "feature-cache")
    cache_manifest = extract_cache(args.dataset, cache, args.checkpoint, stride=5, batch_size=args.vision_batch,
                                   max_episodes=args.max_episodes)
    (out_root / "feature_manifest.json").write_text(json.dumps(cache_manifest, indent=2), encoding="utf-8")
    report_rows = []
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_root = out_root / "models"; model_root.mkdir()
    for horizon in (25, 10):
        records, tail_stats = _collect(cache_manifest, horizon)
        train, eval_rows, train_eps, eval_eps = _prepare(records, horizon)
        if not train or not eval_rows:
            raise RuntimeError(f"H={horizon}: insufficient full windows train={len(train)} eval={len(eval_rows)}")
        for beta in (1e-3, 1e-2, 1e-1):
            seed = args.seed
            model, preds, norm, history = _fit(train, eval_rows, horizon, beta, seed, epochs=args.epochs,
                                               batch_size=args.train_batch, device=device)
            tag = f"H{horizon}_beta{beta:g}"
            model_dir = model_root / tag; model_dir.mkdir()
            torch.save({"state_dict": model.cpu().state_dict(), "normalizers": norm, "history": history}, model_dir / "model.pt")
            model_info = {"checkpoint": str(model_dir / "model.pt"), "seed": seed, "beta": beta,
                          "epochs": args.epochs, "training_episodes": [e["episode_id"] for e in train_eps]}
            eval_input = out_root / "eval_inputs" / tag
            eval_input.parent.mkdir(exist_ok=True)
            _write_eval_input(eval_input, eval_eps, eval_rows, preds, horizon, model_info)
            fm, fs, sm, ss = norm["f_mean"], norm["f_scale"], norm["s_mean"], norm["s_scale"]
            def reconstruct(z, context, model=model, fm=fm, fs=fs, sm=sm, ss=ss, norm=norm, device=device):
                rows = context["row_index"]
                f_raw = context["run"].keys["vlm_img_lang"][rows]
                s_raw = context["proprio"]
                f = torch.from_numpy(((f_raw - fm) / fs).astype(np.float32)).to(device)
                s = torch.from_numpy(((s_raw - sm) / ss).astype(np.float32)).to(device)
                zz = torch.from_numpy(z.astype(np.float32)).to(device)
                with torch.no_grad():
                    y = model.to(device).eval().decode(zz, f, s).cpu().numpy()
                y = y.reshape(len(y), -1) * norm["a_scale"] + norm["a_mean"]
                return y.reshape(len(y), horizon, -1)
            eval_out = out_root / "evaluations" / tag
            eval_out.parent.mkdir(exist_ok=True)
            result = run_all(eval_input, eval_out, horizon=horizon, query_fraction=0.25, seed=seed,
                             mi_config=MIConfig(n_permutations=args.permutations, bootstrap_samples=args.bootstrap,
                                                minimum_groups=10, seed=seed),
                             n_components=32, reconstruct_fn=reconstruct, bootstrap_samples=args.bootstrap,
                             criteria=GoNoGoCriteria(), label=tag)
            result["light_training"] = {"beta": beta, "epochs": args.epochs, "seed": seed,
                                        "train_episodes": len(train_eps), "eval_episodes": len(eval_eps),
                                        "train_rows": len(train), "eval_rows": len(eval_rows),
                                        "tail_stats": tail_stats, "final_epoch": history[-1],
                                        "latent_diagnostics": _latent_diagnostics(preds)}
            (eval_out / "results.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
            report_rows.append((horizon, beta, result))
            del model
    _report(out_root, cache_manifest, report_rows, args)
    print(out_root)
    return out_root


def _latent_diagnostics(preds):
    mu = preds["post_mean"]
    lv = preds["post_logvar"]
    kl = 0.5 * (np.exp(lv) + mu ** 2 - 1 - lv)
    var = mu.var(axis=0)
    return {"kl_post_to_prior_note": "see pipeline MI posterior diagnostic; KL to standard normal below is descriptive",
            "kl_post_to_standard_normal_nats_per_row": float(kl.sum(1).mean()),
            "effective_dims_variance_gt_0.01": int(np.sum(var > 0.01)),
            "z_dim": int(mu.shape[1])}


def _report(out_root, cache_manifest, rows, args):
    lines = ["# Emergent skill light go/no-go", "",
             f"Run: `{out_root}`. Frozen π0.5 base PaliGemma; LIBERO-40 unsegmented task IDs 10–39 only (30 spatial/object/goal tasks).",
             "Task text is passed only to the language-conditioned frozen VLM; task labels and skill annotations are not model inputs.",
             "Whole episodes are split by task: 80% train / 20% held-out evaluation, then existing eval splits held-out initial-state IDs into store/query.",
             "Feature cache samples every 5 environment steps; base and wrist image tokens are mean pooled, in separate language and no-language passes.",
             f"Data episodes cached: {len(cache_manifest['episodes'])}. Training: MLP Enc/Prior/Dec, z_dim=32, AdamW, normalized action MSE + β·analytic KL(q||p), {args.epochs} epochs, seed {args.seed}.",
             "The evaluation pipeline and its i–iii thresholds are unchanged. `memory z` uses its fixed PCA key and relative-distance threshold from the existing evaluator; cross-reconstruction calls this trained decoder.", "",
             "| H | β | train episodes / rows | eval episodes / rows | tail windows dropped | KL(post‖N(0,I)) nats/row | effective dims | go/no-go | i | ii | iii |",
             "|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|"]
    for h, beta, result in rows:
        d = result["light_training"]; c = result["gonogo"]["criteria"]
        lines.append(f"| {h} | {beta:g} | {d['train_episodes']} / {d['train_rows']} | {d['eval_episodes']} / {d['eval_rows']} | "
                     f"{d['tail_stats']['tail_windows_dropped']} | {d['latent_diagnostics']['kl_post_to_standard_normal_nats_per_row']:.4g} | "
                     f"{d['latent_diagnostics']['effective_dims_variance_gt_0.01']}/32 | {result['gonogo']['overall']} | "
                     f"{c['i_prior_mi']['status']} | {c['ii_crosstask_gap']['status']} | {c['iii_cross_reconstruction']['status']} |")
    lines += ["", "## Caveats", "",
              "This is a small-model decision aid, not a VLA result. The decoder predicts continuous action vectors with an MLP, not the production flow-matching action expert. The posterior KL-to-standard-normal and effective-dimension count are descriptive; go/no-go uses the existing evaluator unchanged.",
              "Quaternion/position/gripper motion labels are derived from saved LIBERO state trajectories and remain evaluation-only. No skill ID or annotation columns are read.",
              "The legacy crosstask reader requires a skill plan. Generated NPZs contain a synthetic `[pick, place]` proxy plan, and its temporal split uses only the executed gripper-open/close trace. No annotation columns are read; these are not LIBERO's annotated subtasks. Only the different-task condition is treated as meaningful evidence for criterion (ii).",
              "See each `evaluations/H*_beta*/report.md` and `results.json` for criterion evidence, MI, cross-task gaps, and reconstruction errors.", ""]
    (out_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUTS)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--vision-batch", type=int, default=2)
    p.add_argument("--train-batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--permutations", type=int, default=100)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
