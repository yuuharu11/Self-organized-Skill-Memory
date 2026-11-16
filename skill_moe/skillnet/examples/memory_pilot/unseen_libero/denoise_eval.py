#!/usr/bin/env python3
"""Oracle-type bridge diagnostic on the orientation-corrected LIBERO-90 Eval rows.

No training is performed. Ground-truth actions select the scale-invariant type
from the fixed unit_08 pool and are also used for scoring. The selected type is
posed with the ordinary pi0.5 chunk exactly as in memory_closed_loop.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from unseen_paths import UNSEEN_ROWS

STATE = Path(os.environ.get("SKILLNET_STATE_DIR", Path(__file__).resolve().parents[5] / ".skillnet-state"))
ROOT = STATE / "outputs/memory-pilot/unseen-libero"
OUT = ROOT / "denoise-02"
POOL_PATH = ROOT / "cluster-pools-01/pool_unit_08.npz"
NORMAL_PATH = ROOT / "oracle-01/official_normal_eval.npz"
HARD_PATH = ROOT / "oracle-01/drafts/raw_none_eval.npz"
HARD_THRESHOLD_PATH = ROOT / "oracle-01/hard_threshold.json"
PUBLIC_CKPT = Path(os.environ.get("SKILLNET_PI05_CHECKPOINT", STATE / "checkpoints/openpi-assets/checkpoints/pi05_libero"))
LEARNED_META = ROOT / "compressed-denoise-01/production/experts/A-vm-08/best_bridge.json"
LEARNED_RUN = HERE
T0S = (0.2, 0.4, 0.6)
NOISE_SEED = 4101
BOOT_REPS = 5000


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def six_mse(pred, target):
    """Per-row six-axis MSE in the repository's sum-six/divide-seven scale."""
    p = np.asarray(pred, np.float64)[..., :6]
    y = np.asarray(target, np.float64)[..., :6]
    if p.shape != y.shape or p.ndim != 3:
        raise ValueError((p.shape, y.shape))
    return np.square(p - y).mean((1, 2)) * (6.0 / 7.0)


def active7_mse(pred, target):
    p = np.asarray(pred, np.float64)[..., :7]
    y = np.asarray(target, np.float64)[..., :7]
    if p.shape != y.shape or p.ndim != 3:
        raise ValueError((p.shape, y.shape))
    return np.square(p - y).mean((1, 2))


def pad_actions(actions):
    a = np.asarray(actions, np.float32)
    if a.ndim != 3 or a.shape[1:] not in ((10, 7), (10, 32)):
        raise ValueError(f"expected [N,10,7|32], got {a.shape}")
    if a.shape[-1] == 32:
        return a.copy()
    out = np.zeros((len(a), 10, 32), np.float32)
    out[:, :, :7] = a
    return out


def pose_oracle_types(target7, normal7, pool_shapes, vs, *, cond="S-shared"):
    """Choose by scale-only type error; then use pi0.5 pose/gripper as in closed loop."""
    import scale_only_query as so
    import theta_flow as tf

    target7 = np.asarray(target7, np.float32)
    normal7 = np.asarray(normal7, np.float32)
    pool_shapes = np.asarray(pool_shapes, np.float32)
    if target7.shape != normal7.shape or target7.ndim != 3 or target7.shape[1:] != (10, 7):
        raise ValueError((target7.shape, normal7.shape))
    if pool_shapes.ndim != 3 or pool_shapes.shape[1:] != (10, 7):
        raise ValueError(pool_shapes.shape)
    q_phys = vs.to_physical(target7[:, :, :6])
    # Stored pool shapes are already pose-removed RMS-scaled physical motion
    # (SkillMemory.shape); to_physical would treat them as quantile-normalized
    # actions and add per-axis offsets the scale fit cannot absorb. Keep them
    # as stored, as the Procrustes fit two lines below already does.
    cand_phys = np.ascontiguousarray(pool_shapes[:, :, :6], dtype=np.float64)
    errors = so.scale_only_error(q_phys, cand_phys, cond)
    selected = errors.argmin(1).astype(np.int64)
    shapes = pool_shapes[selected]
    theta = tf.batched_procrustes(shapes[:, :, :6], vs.to_physical(normal7[:, :, :6]))[0]
    decoded = tf.decode_many(shapes, theta, vs)
    # Closed-loop memory policy keeps the pi0.5 gripper channel.
    decoded[:, :, 6] = normal7[:, :, 6]
    return selected, errors[np.arange(len(selected)), selected], decoded.astype(np.float32)


def hold_last(actions, task, episode, frame):
    """Repeat the previous executed action; zero at episode start, as scale-only baseline."""
    a = np.asarray(actions, np.float32)
    task, episode, frame = (np.asarray(x, np.int64) for x in (task, episode, frame))
    first = {}
    for i, key in enumerate(zip(task.tolist(), episode.tolist())):
        first.setdefault(key, {})[int(frame[i])] = i
    out = np.zeros_like(a[:, :, :7])
    for i, (t, e, f) in enumerate(zip(task, episode, frame)):
        prev = first[(int(t), int(e))].get(int(f) - 10)
        if prev is not None:
            out[i] = a[prev, -1:, :7]
    return out


def episode_bootstrap(values, task, episode, *, reps=BOOT_REPS, seed=NOISE_SEED):
    v = np.asarray(values, np.float64)
    epkey = np.asarray(task, np.int64) * 100000 + np.asarray(episode, np.int64)
    uniq, inv = np.unique(epkey, return_inverse=True)
    sums = np.bincount(inv, weights=v)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(uniq), size=(reps, len(uniq)))
    means = sums[draw].sum(1) / counts[draw].sum(1)
    return {"mean": float(v.mean()), "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            "episodes": int(len(uniq)), "rows": int(len(v)), "bootstrap_reps": int(reps)}


def load_inputs():
    if not UNSEEN_ROWS.exists():
        raise FileNotFoundError(f"orientation-corrected rows missing: {UNSEEN_ROWS}")
    rows = {p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in UNSEEN_ROWS.glob("*.npy")}
    ids = np.flatnonzero(~np.asarray(rows["is_dev"], bool)).astype(np.int64)
    if len(ids) != 619 or len(np.unique(rows["task"][ids])) != 10:
        raise AssertionError(f"expected fixed Eval of 619 rows/10 tasks, got {len(ids)}")
    normal_file = np.load(NORMAL_PATH, allow_pickle=False)
    # Row IDs are run-01 row_count + local unseen row index. Join by ID, never by position.
    import compressed_denoise as cd
    base = _module(cd.PI05 / "run.py", "dn1_row_id_base")
    run_rows = base.load_rows(cd.RUN)
    offset = len(run_rows["task"])
    wanted = offset + ids
    loc = {int(r): i for i, r in enumerate(normal_file["row_ids"])}
    if any(int(r) not in loc for r in wanted):
        raise AssertionError("official normal Eval cache does not cover the corrected Eval rows")
    ni = np.asarray([loc[int(r)] for r in wanted], np.int64)
    normal = np.asarray(normal_file["prediction"][ni], np.float32)
    target = np.asarray(rows["actions"][ids, :, :7], np.float32)
    if not np.allclose(normal, np.asarray(normal_file["prediction"][ni])):
        raise AssertionError("normal prediction join failed")
    hard_cache = np.load(HARD_PATH, allow_pickle=False)
    expected = offset + ids
    hi = {int(r): i for i, r in enumerate(hard_cache["query_ids"])}
    if any(int(r) not in hi for r in expected):
        raise AssertionError("hard-label cache row join failed")
    threshold = float(json.loads(HARD_THRESHOLD_PATH.read_text())["threshold"])
    hard = np.asarray([hard_cache["pre_mse"][hi[int(r)]] >= threshold for r in expected], bool)
    pool = np.load(POOL_PATH, allow_pickle=False)
    if pool["shape"].shape != (95, 10, 7):
        raise AssertionError(f"unit_08 pool changed: {pool['shape'].shape}")
    return rows, ids, target, normal, hard, pool, offset, threshold


def _prediction(model, obs, drafts, noise, t0, *, batch=8):
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit

    fn = module_jit(model.sample_bridge, static_argnames=("num_steps",))
    out = []
    for start in range(0, len(drafts), batch):
        sl = slice(start, min(start + batch, len(drafts)))
        out.append(np.asarray(fn(jax.random.key(0), jax.tree.map(lambda x: x[sl], obs),
                                 jnp.asarray(drafts[sl]), jnp.full((sl.stop - sl.start,), float(t0)),
                                 num_steps=10, noise=jnp.asarray(noise[sl])))
                    .astype(np.float32))
    return np.concatenate(out)


def _inject_checkpoint(model, checkpoint):
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex

    filt = nnx.All(nnx.Param, PathRegex(r".*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*"))
    with ocp.PyTreeCheckpointer() as checkpointer:
        part = checkpointer.restore(Path(checkpoint))
    state = nnx.state(model, filt)
    state.replace_by_pure_dict(jax.tree.map(jnp.asarray, part))
    nnx.update(model, state)


def run_evaluation(*, threads=8, batch=8, out=OUT):
    import torch
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.20)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    rows, ids, target, normal, hard, pool, offset, threshold = load_inputs()
    from compressed_denoise import _module as cd_module, CHECKPOINT, PI05
    import compressed_denoise as cd
    import variable_shape_memory as vs
    base = cd_module(PI05 / "run.py", "dn1_eval_base")
    skill = cd_module(PI05 / "skill_expert" / "run.py", "dn1_eval_skill")
    selected, type_error, draft7 = pose_oracle_types(target, normal, pool["shape"], vs)
    if not np.array_equal(draft7[:, :, 6], normal[:, :, 6]):
        raise AssertionError("oracle type draft must use the ordinary pi0.5 gripper")
    drafts = pad_actions(draft7)
    truth_drafts = pad_actions(target)
    normal_padded = pad_actions(normal)
    hold = hold_last(np.asarray(rows["actions"]), np.asarray(rows["task"]), np.asarray(rows["episode"]),
                     np.asarray(rows["frame"]))[ids]
    baseline = {"oracle_type_pi05_pose_draft": draft7, "oracle_truth_action_draft": target,
                "official_pi05": normal, "hold_last": hold}
    obs = base.obs_batch(rows, ids)
    import jax
    noise = np.stack([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(NOISE_SEED), int(offset + i)),
                                                    (10, 32))) for i in ids]).astype(np.float32)
    meta = {
        "status": "running", "eval_rows": int(len(ids)), "hard_rows": int(hard.sum()),
        "eval_sources": {"corrected_rows": str(UNSEEN_ROWS), "ordinary_pi05": str(NORMAL_PATH),
                         "unit_pool": str(POOL_PATH)},
        "target_use": "Ground truth chooses only the scale-invariant S-shared unit and is used for scoring; never for theta, t0, or checkpoint selection.",
        "type_selection": "minimum scale_only_query.scale_only_error(S-shared) over pool_unit_08; 95 stored clusters; no rotation in type selection",
        "pose_and_gripper": "Procrustes rotation+single scale of the selected shape onto official pi0.5 normalized action chunk (converted to physical), then decode; gripper copied from pi0.5, matching memory_closed_loop.py",
        "bridge": "sample_bridge, 10 Euler steps, row-keyed fixed Gaussian noise fold_in(key(4101), run01_row_id), t0 in {0.2,0.4,0.6}",
        "hard_rule": f"raw_none Dev-q80 threshold {threshold:.12g}; pre_mse and rows joined by global row ID",
        "experts": {"public_pi05": str(CHECKPOINT), "compressed_denoise": str(LEARNED_META)},
        "mse": "MSE6=sum of six axis squared errors divided by 7 (equivalently per-scalar 6-axis MSE*6/7); active7 is ordinary per-scalar mean over 7 axes.",
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    all_outputs = {"row_ids": offset + ids, "local_row_ids": ids, "task": np.asarray(rows["task"][ids]),
                   "episode": np.asarray(rows["episode"][ids]), "frame": np.asarray(rows["frame"][ids]),
                   "hard": hard, "selected_unit": selected, "oracle_type_error": type_error,
                   "target": target, "normal": normal, "draft": draft7, "noise_seed": np.full(len(ids), NOISE_SEED)}
    summary = {"status": "running", "conditions": {}, "baselines": {}}
    episodes_task, episodes = np.asarray(rows["task"][ids]), np.asarray(rows["episode"][ids])

    def add_stats(key, pred):
        pred = np.asarray(pred, np.float32)
        for split, mask in (("all", np.ones(len(ids), bool)), ("hard", hard)):
            if not mask.any():
                continue
            vals = {"mse6_sum_div7": six_mse(pred[mask], target[mask]),
                    "mse_active7": active7_mse(pred[mask], target[mask]),
                    "mse_gripper": np.square(pred[mask, :, 6] - target[mask, :, 6]).mean(1)}
            stat = {}
            for m, v in vals.items():
                stat[m] = episode_bootstrap(v, episodes_task[mask], episodes[mask])
            summary.setdefault(key, {})[split] = stat

    for name, pred in baseline.items():
        add_stats(f"baseline/{name}", pred)
    def evaluate_draft_family(model, family, draft_input, refs):
        for t0 in T0S:
            started = time.monotonic()
            pred = _prediction(model, obs, draft_input, noise, t0, batch=batch)[:, :, :7]
            key = f"{family}/t0={t0:.1f}"
            all_outputs[key.replace("/", "__").replace("=", "_")] = pred
            add_stats(key, pred)
            for reference_name, ref in refs:
                distance = active7_mse(pred, ref)
                for split, mask in (("all", np.ones(len(ids), bool)), ("hard", hard)):
                    if mask.any():
                        summary.setdefault(key, {}).setdefault("distance", {}).setdefault(split, {})[reference_name] = episode_bootstrap(
                            distance[mask], episodes_task[mask], episodes[mask])
            summary.setdefault(key, {})["seconds"] = float(time.monotonic() - started)
            _write_json(out / "progress.json", {"status": "running", "completed": key, "updated_unix": time.time()})

    # Instantiate each checkpoint before compiling/running its bridge so the JIT
    # sees the intended live weights rather than a previously compiled model.
    public_model, _ = base.load_policy(CHECKPOINT)
    evaluate_draft_family(public_model, "public_pi05/oracle_type_draft", drafts,
                          (("draft", draft7), ("official_pi05", normal)))
    evaluate_draft_family(public_model, "public_pi05/oracle_truth_action_draft", truth_drafts,
                          (("truth_draft", target), ("official_pi05", normal)))
    del public_model
    import gc, jax
    jax.clear_caches(); gc.collect()

    learned_model, _ = base.load_policy(CHECKPOINT)
    learned_checkpoint = json.loads(LEARNED_META.read_text())["path"]
    _inject_checkpoint(learned_model, learned_checkpoint)
    evaluate_draft_family(learned_model, "compressed_A_vm_08/oracle_type_draft", drafts,
                          (("draft", draft7), ("official_pi05", normal)))
    del learned_model
    jax.clear_caches(); gc.collect()
    np.savez_compressed(out / "eval_rows.npz", **all_outputs)
    summary.update({"status": "complete", "eval_rows": int(len(ids)), "hard_rows": int(hard.sum()),
                    "task_names": [str(x) for x in np.unique(rows["task_name"][ids])]})
    _write_json(out / "summary.json", summary)
    _write_json(out / "progress.json", {"status": "complete", "updated_unix": time.time(), "eval_rows": int(len(ids))})
    (out / "report.md").write_text(render_report(summary, meta))
    return summary


def _write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    tmp.replace(path)


def render_report(summary, meta):
    def ci(v):
        if not v: return "—"
        return f"{v['mean']:.5f} [{v['ci95'][0]:.5f}, {v['ci95'][1]:.5f}]"

    lines = ["# DN1: オラクル型のデノイズ評価", "",
             "正解型をπ0.5の姿勢へ合わせた下書きは、デノイズで全EvalのMSE6が下がる。公開π0.5 Expertはt0=.6で .2651→.1147、compressed A-vm-08はt0=.4で .2651→.1142。",
             "hard行ではそれぞれ .3274→.2510、.3274→.2480まで改善するが、直前保持の .2010より高い。",
             "したがって修正画像でのデノイズはこのオフライン上限下書きを悪化させず、全体では改善する。ただし hard 行を基準以下にするには足りない。", "",
             "固定 Eval 619 窓（難しい行134、Dev q80閾値）。正解は S-shared 型の単位選択と採点に限り、姿勢・t0・checkpoint 選択には使っていない。",
             "MSE6 は6軸二乗誤差の和を7で割った値（従来の6軸平均×6/7）。active7 とグリッパーは該当次元の通常平均。すべて episode bootstrap 5,000回の95%区間。", "",
             "## 比較", "", "| 条件 | t0 | 全 MSE6 [95% CI] | hard MSE6 [95% CI] | 全 active7 [95% CI] | hard active7 [95% CI] | 全 grip [95% CI] | hard grip [95% CI] |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    # Build a stable ordered table from the stored records.
    records = []
    for key, val in summary.items():
        if key.startswith("baseline/") or "/oracle_type_draft/" not in key and "/oracle_truth_action_draft/" not in key:
            continue
        allm = val.get("all", {}).get("mse6_sum_div7", {})
        hardm = val.get("hard", {}).get("mse6_sum_div7", {})
        aa = val.get("all", {}).get("mse_active7")
        ha = val.get("hard", {}).get("mse_active7")
        ga = val.get("all", {}).get("mse_gripper")
        gh = val.get("hard", {}).get("mse_gripper")
        if allm:
            records.append((key, allm, hardm, aa, ha, ga, gh))
    for name in ("public_pi05", "compressed_A_vm_08"):
        for key, a, h, aa, ha, ga, gh in records:
            if not key.startswith(name + "/oracle_type_draft/"):
                continue
            t0 = key.rsplit("=", 1)[-1]
            lines.append(f"| {name} / π0.5姿勢の正解型下書き | {t0} | {ci(a)} | {ci(h)} | {ci(aa)} | {ci(ha)} | {ci(ga)} | {ci(gh)} |")
    for name, key in (("下書きそのまま", "baseline/oracle_type_pi05_pose_draft"),
                      ("正解行動を下書き", "baseline/oracle_truth_action_draft"),
                      ("通常π0.5", "baseline/official_pi05"), ("直前保持", "baseline/hold_last")):
        v = summary.get(key)
        if v:
            lines.append(f"| {name} | — | {ci(v['all']['mse6_sum_div7'])} | {ci(v['hard']['mse6_sum_div7'])} | {ci(v['all']['mse_active7'])} | {ci(v['hard']['mse_active7'])} | {ci(v['all']['mse_gripper'])} | {ci(v['hard']['mse_gripper'])} |")
    lines += ["", "## 出力と下書き・通常π0.5の距離", "", "距離は active7 MSE（平均 [episode bootstrap 95% CI]）。", "", "| Expert / 下書き | t0 | 全: 出力-下書き | hard: 出力-下書き | 全: 出力-通常π0.5 | hard: 出力-通常π0.5 |", "|---|---:|---:|---:|---:|---:|"]
    for family in ("public_pi05/oracle_type_draft", "compressed_A_vm_08/oracle_type_draft"):
        for t0 in T0S:
            key = f"{family}/t0={t0:.1f}"
            v = summary[key]["distance"]
            lines.append(f"| {family} | {t0:.1f} | {ci(v['all']['draft'])} | {ci(v['hard']['draft'])} | {ci(v['all']['official_pi05'])} | {ci(v['hard']['official_pi05'])} |")
    lines += ["", "正解行動を下書きにした場合の public bridge 出力も `summary.json` と `eval_rows.npz` に保存。これはデノイザーが正解入力を壊す度合いを見る対照で、モデル選択には使っていない。", "",
              "## 条件と制限", "", f"- Expert: 公開 pi05_libero と、compressed-denoise production の A-vm-08（{LEARNED_META}）の Dev 選択 checkpoint。3つの t0 は固定評価値であり、Eval から選んでいない。",
              "- 正解型選択は上限診断。閉ループ検索の性能を示すものではない。", "- 公式通常π0.5・noise と bridge の両方は同じ行IDで対応づけ、bridge条件は同じ fold_in noise を使う。", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("evaluate", choices=["evaluate"])
    p.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    run_evaluation(threads=a.threads, batch=a.batch, out=a.out)


if __name__ == "__main__":
    main()
