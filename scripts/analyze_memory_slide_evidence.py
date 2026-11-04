"""CPU-only audit and task-held-out routing of saved lightweight FP predictions.

No simulator, policy inference, backbone training, or source-file mutation.
The historical experiment's instruction condition is the primary evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

NORMAL = "N:noise:n10"
MEMORY = "A:unit:cal_sim"
METHODS = (NORMAL, "N:unit:cal_sim", MEMORY, "N:raw:cal_sim",
           "A:shuf:cal_sim", "A:gold_unit:cal_sim", "A:noise:n10")
TRAIN = {"A": tuple(range(20, 40)), "B": tuple(range(10, 30))}
EVAL = {"A": tuple(range(10, 20)), "B": tuple(range(30, 40))}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metadata(cache):
    """Recreate historical window order without loading images or VLM features."""
    manifest = json.loads((cache / "manifest.json").read_text())
    columns = {k: [] for k in ("episode", "task", "frame")}
    for ep in manifest["episodes"]:
        with np.load(ep["cache"], allow_pickle=False) as z:
            task = int(z["task_index"])
            if task not in range(10, 40):
                continue
            steps = z["step"].astype(int)
            actions, states = z["action"], z["env_state"]
            env_step = z["env_step"] if "env_step" in z else np.arange(len(states))
            lookup = {int(s): i for i, s in enumerate(env_step)}
            for step in steps:
                start = lookup.get(int(step), -1)
                if start >= 0 and start + 25 < len(states) and start + 25 <= len(actions):
                    columns["episode"].append(str(ep["episode_id"]))
                    columns["task"].append(task)
                    columns["frame"].append(int(step))
    return {k: np.asarray(v) for k, v in columns.items()}


def subset(meta, mask):
    return {k: v[mask] for k, v in meta.items()}


def heldout_metadata(meta, split, seed, per_task):
    train = subset(meta, np.isin(meta["task"], TRAIN[split]))
    rng, held = np.random.default_rng(seed + 3), set()
    for task in TRAIN[split]:
        episodes = np.unique(train["episode"][train["task"] == task])
        if len(episodes) > per_task + 3:
            held.update(map(str, rng.choice(episodes, per_task, replace=False)))
    return subset(train, np.isin(train["episode"], list(held)))


def check_alignment(meta, task):
    if len(meta["task"]) != len(task) or not np.array_equal(meta["task"], task):
        raise ValueError("saved predictions and reconstructed window metadata do not align")
    pairs = list(zip(meta["episode"], meta["frame"].tolist()))
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate episode/frame windows")


def balanced_weights(tasks, episodes):
    """Equal task, then equal episode within task, then equal windows."""
    w = np.zeros(len(tasks), dtype=float)
    for task in np.unique(tasks):
        task_mask = tasks == task
        names = np.unique(episodes[task_mask])
        for ep in names:
            mask = task_mask & (episodes == ep)
            w[mask] = 1.0 / (len(np.unique(tasks)) * len(names) * mask.sum())
    return w


def episode_blocks(values, meta):
    return {str(task): np.asarray([
        float(np.mean(values[(meta["task"] == task) & (meta["episode"] == ep)]))
        for ep in np.unique(meta["episode"][meta["task"] == task])])
        for task in np.unique(meta["task"])}


def hierarchical_interval(values, meta, draws, seed):
    blocks = list(episode_blocks(values, meta).values())
    point = float(np.mean([x.mean() for x in blocks]))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        means = []
        for i in rng.integers(len(blocks), size=len(blocks)):
            b = blocks[i]
            means.append(b[rng.integers(len(b), size=len(b))].mean())
        samples.append(np.mean(means))
    lo, hi = np.quantile(samples, [.025, .975])
    return {"task_episode_equal_mean": point, "ci95": [float(lo), float(hi)]}


def evaluate_gate(features, tasks, episodes, normal, memory):
    """Outer leave-one-task-out; no held-out labels in scaler/model/threshold."""
    if not np.isfinite(features).all():
        raise ValueError("non-finite runtime features")
    n = len(tasks)
    choices = {k: np.zeros(n, dtype=bool) for k in ("constant", "threshold", "logistic")}
    folds = []
    for task in np.unique(tasks):
        test = tasks == task
        train = ~test
        if not train.any():
            raise ValueError("routing needs at least two distinct tasks")
        if set(episodes[train]) & set(episodes[test]):
            raise ValueError("episode leakage across held-out task")
        x, a, b = features[train], normal[train], memory[train]
        weights = balanced_weights(tasks[train], episodes[train])
        choose_mem = bool(np.sum(weights * b) < np.sum(weights * a))
        choices["constant"][test] = choose_mem
        thresholds = np.r_[-np.inf, np.quantile(x[:, 0], np.linspace(0, 1, 21)), np.inf]
        costs = [np.sum(weights * np.where(x[:, 0] >= t, b, a)) for t in thresholds]
        threshold = float(thresholds[int(np.argmin(costs))])
        choices["threshold"][test] = features[test, 0] >= threshold
        labels = (b < a).astype(int)
        if len(np.unique(labels)) == 1:
            choices["logistic"][test] = bool(labels[0])
        else:
            scaler = StandardScaler().fit(x, sample_weight=weights)
            # Cost-sensitive classification: a wrong choice costs |b-a|.
            sample_weight = weights * np.abs(a - b)
            sample_weight *= len(sample_weight) / max(sample_weight.sum(), 1e-12)
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=0)
            model.fit(scaler.transform(x), labels, sample_weight=sample_weight)
            choices["logistic"][test] = model.predict(scaler.transform(features[test])).astype(bool)
        folds.append({"test_task": int(task), "train_tasks": list(map(int, np.unique(tasks[train]))),
                      "train_episodes": int(len(np.unique(episodes[train]))),
                      "test_episodes": int(len(np.unique(episodes[test]))),
                      "train_rows": int(train.sum()), "test_rows": int(test.sum()),
                      "threshold": threshold if np.isfinite(threshold) else str(threshold),
                      "constant_memory": choose_mem})
    errors = {k: np.where(choice, memory, normal) for k, choice in choices.items()}
    errors.update({"normal": normal, "memory": memory,
                   "oracle": np.minimum(normal, memory), "random50_expected": .5 * (normal + memory)})
    return choices, errors, folds


def summarize_seed_arrays(arrays, metas, draws, seed):
    """Seeds share evaluation observations; average seeds before cluster bootstrap."""
    same = all(np.array_equal(metas[0][k], m[k]) for m in metas[1:] for k in metas[0])
    means = [float(x.mean()) for x in arrays]
    out = {"row_mean_seed_average": float(np.mean(means)),
           "seed_sd": float(np.std(means, ddof=1)), "per_seed_row_mean": means,
           "windows_per_seed": [len(a) for a in arrays],
           "episodes_per_seed": [len(np.unique(m["episode"])) for m in metas],
           "tasks_per_seed": [len(np.unique(m["task"])) for m in metas]}
    if same:
        out.update(hierarchical_interval(np.mean(arrays, axis=0), metas[0], draws, seed))
        out["ci_scope"] = "task/episode bootstrap after averaging 3 seeds on shared observations"
    else:
        # D1 episode membership varies by model seed: show equal task/episode point only.
        out["task_episode_equal_mean"] = float(np.mean([
            np.mean([b.mean() for b in episode_blocks(x, m).values()]) for x, m in zip(arrays, metas)]))
        out["ci95"] = None
        out["ci_scope"] = "no pooled CI: held-out episodes vary across model seeds"
    return out


def fmt(x):
    return f"{x:.4f}"


def write_report(out, payload):
    lines = ["# スキルメモリ発表用：既存データの充足性と追加CPU診断", "",
             "軽量FPの同一実験を基盤に、goal単独、検索、保存内容、二経路の切替を再評価した。",
             "新規の環境rollout、GPU推論、VLA/FP再学習は行っていない。新しい学習は小さなCPUルーターのみ。", "",
             "## 1. 評価の契約", "",
             "- 行動MSEは元の行動単位で8ノイズ×25step×7次元を平均したもの。実演観測でのオフライン評価。",
             "- 主条件はinstruction由来のサブタスク。goldサブタスク条件と混ぜない。",
             "- A: object+spatialで学習しgoalを評価。B: object+goalで学習しspatialを評価。",
             "- D1: 既知タスクの保留エピソード。D3: 対象suiteをメモリに追加しない。D2: 先行実演の逐次追加。",
             "- 新しい区間ラベルは付与していない。検索の正解は将来の正解行動を符号化したコードに機能的に最も近い単位であり、人間の意味ラベルではない。",
             "- 重なる窓を独立標本として扱わない。区間推定はtask→episodeで再標本化し、3 model seedを先に平均する。seed不確実性全体の区間ではない。", "",
             "## 2. データ量とgoal単独の評価", "",
             "表のMSEは過去の表と同じ窓平均→seed平均。窓数はseed 0の数で、独立試行数ではない。", "",
             "| 評価集合 | タスク数 | エピソード数 | 窓数 | 通常FP | 同じFP・運動開始 | 運動経路も学習 | 生系列検索 | 正解単位の参考条件 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, cell in payload["cohorts"].items():
        b = cell["methods"]
        stat = b[NORMAL]
        vals = [b[m]["row_mean_seed_average"] for m in (NORMAL, "N:unit:cal_sim", MEMORY,
                                                          "N:raw:cal_sim", "A:gold_unit:cal_sim")]
        lines.append(f"| {key} | {stat['tasks_per_seed'][0]} | {stat['episodes_per_seed'][0]} | {stat['windows_per_seed'][0]} | " + " | ".join(map(fmt, vals)) + " |")
    lines += ["", "同じ生成器での保存内容比較は下表で固定t0=0.4、n=4に揃える。上表の生系列検索と追加学習済み単位検索を直接引いて、圧縮単位の寄与とは解釈しない。", "",
              "## 3. 保持・汎化と検索の診断", "",
              "| 保留suite・追加なし | 通常→運動開始の窓平均差 | task/episode等重みの差 [95%区間] | top1正解単位一致率 | 取得単位→正解単位のMSE |",
              "|---|---:|---|---:|---|"]
    for split in ("A", "B"):
        c = payload["cohorts"][f"{split}_D3_all"]
        delta = c["memory_minus_normal"]
        ci = delta["ci95"]
        lines.append(f"| {split} | {delta['row_mean_seed_average']:+.4f} | {delta['task_episode_equal_mean']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {c['strict_top1_mean']:.1%} | {c['methods'][MEMORY]['row_mean_seed_average']:.4f} → {c['methods']['A:gold_unit:cal_sim']['row_mean_seed_average']:.4f} |")
    lines += ["", "正解単位条件は実行不能な参考診断であり、最良出力を与える単位を全探索した厳密な上限ではない。一致率が低いだけから失敗原因を検索に限定しない。", "",
              "## 4. 保存するもの：同じFP・t0=0.4・n=4", "",
              "既存結果JSONから抽出したseed平均。個々の固定t0予測は保存されていないため、新しいepisode bootstrapは行わない。", "",
              "| 条件 | 圧縮した運動単位 | 生の行動系列 | 単位−生系列 |",
              "|---|---:|---:|---:|"]
    for key, cell in payload["fixed_budget_contents"].items():
        lines.append(f"| {key} | {cell['unit']:.4f} | {cell['raw']:.4f} | {cell['unit']-cell['raw']:+.4f} |")
    lines += ["", "## 5. 追加実験：観測から得られる信頼度で二経路を選ぶ", "",
              "通常FPと追加学習済みメモリ経路を選ぶ。外側はleave-one-task-out。test taskの全episodeを学習から除外し、標準化・閾値の選択・CPUロジスティック回帰をtrain taskだけで行う。",
              "入力は検索類似度・検索重みエントロピーの2値とその二次項。未来行動、誤差、task ID、episode ID、gold単位を推論入力にしない。教師は保存済み行動誤差の差。",
              "thresholdはtrainで21分位点と常時選択の候補から選ぶ。logisticはC=1固定、誤選択の損失|誤差差|で重み付け。各タスク・各episodeの重みを均等にする。",
              "これは既に分析した10タスク上の探索的cross-validationで、未使用の最終テストではない。予測は既存両経路の誤差から選択するため、新たな動作・成功率を測っていない。", "",
              "| 分割 | 経路選択 | 窓平均MSE | task/episode等重みMSE | 通常との差 [95%区間] | 常時メモリとの差 [95%区間] | メモリ選択率 |",
              "|---|---|---:|---:|---|---|---:|"]
    for split, routing in payload["routing"].items():
        for name, cell in routing["methods"].items():
            ci = cell["minus_normal"]["ci95"]
            mem_ci = cell["minus_memory"]["ci95"]
            lines.append(f"| {split} | {name} | {cell['row_mean_seed_average']:.4f} | {cell['task_episode_equal_mean']:.4f} | {cell['minus_normal']['task_episode_equal_mean']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {cell['minus_memory']['task_episode_equal_mean']:+.4f} [{mem_ci[0]:+.4f}, {mem_ci[1]:+.4f}] | {cell.get('memory_choice_rate', float('nan')):.1%} |")
    lines += ["", "oracleは将来の正解誤差で良い経路を選ぶ実行不能な下限。random50は独立なランダム選択の期待値。constantは訓練タスクだけで常時使う経路を選ぶ対照。ルーターは少なくともこのconstant対照とも比較する。",
              "両経路の誤差を既存8ノイズの平均から選択している。単一生成サンプルのルーティングや共有計算での実行時間削減は検証していない。", "",
              "## 6. 追加で必要なもの・今回の限界", "",
              "- goal単独とtask/episode別の再集計、同じFP・同じ計算予算の保存内容比較、信頼度ルーターのtask-held-out評価は今回補完した。",
              "- 現在の3 seed・各保留10タスクは探索的な発表材料には使えるが、多数の重なる窓から広範な汎化を主張しない。",
              "- 同じ軽量実験のtop-k順位・単位IDの全行記録、decoder/検索器の重みは出力先に残っていない。top5/top20の再測定は今回の出力だけではできない。別実験のhit率をこの表へ混ぜない。",
              "- 同じ生成器での生系列対単位には保存件数・bytesの差が残る。圧縮の効率性を主張するなら、同じメモリ容量の対照が必要。",
              "- 同じ軽量集合で、独立ノイズの通常FP二本を選ぶ対照は未補完。8サンプルそれぞれの全行出力がないため、既存の平均誤差からは再計算できない。メモリ固有の二経路利得を主張する段階では必要。",
              "- b+rの分離、単位の意味的解釈、新しい運動の獲得、閉ループ成功、接触での頑健性はこの再集計だけでは確立しない。",
              "- 小ルーターの学習は過去の政策重みを変更しない。A/DMP/新expertの学習・環境走行は再開していない。", "",
              "## 再現と出典", "", f"- Source: `{payload['source']}`", f"- 実行コマンド: `{payload['command']}`",
              "- 入力hash・照合数・fold監査はsummary.json、全タスク別値はtask_metrics.csv、各foldの選択はrouting_predictions.npz。",
              "- 元のresults.json/per_row.npz/cacheは読み取りのみ。"]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    history = json.loads((args.source / "results.json").read_text())
    arrays = np.load(args.source / "per_row.npz", allow_pickle=False)
    metadata = load_metadata(args.cache)
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "source": str(args.source),
               "command": " ".join(__import__("sys").argv),
               "resources": {"cpu_threads": 1, "gpu": False, "priority": "nice19"},
               "source_hashes": {name: sha256(args.source / name) for name in ("results.json", "per_row.npz")},
               "cache_manifest_sha256": sha256(args.cache / "manifest.json"),
               "cohorts": {}, "routing": {}, "fixed_budget_contents": {},
               "notes": {"bootstrap_draws": args.bootstrap, "source_condition": "instruction"}}
    metric_rows, prediction_arrays = [], {}
    for split in ("A", "B"):
        new_meta = subset(metadata, np.isin(metadata["task"], EVAL[split]))
        for cond in ("D1", "D3", "D2"):
            groups = {"all": None}
            if cond == "D1" and split == "B":
                groups.update({"goal_only": tuple(range(10, 20)), "object_only": tuple(range(20, 30))})
            for group, tasks in groups.items():
                key = f"{split}_{cond}_{group}"
                mets, metas, hits = {name: [] for name in METHODS}, [], []
                for seed in (0, 1, 2):
                    root = f"{split}_s{seed}"
                    meta = heldout_metadata(metadata, split, seed, history["holdout_episodes_per_task"]) if cond == "D1" else new_meta
                    check_alignment(meta, arrays[f"{root}|{'D1_task' if cond == 'D1' else 'new_task'}"])
                    mask = np.ones(len(meta["task"]), bool) if tasks is None else np.isin(meta["task"], tasks)
                    metas.append(subset(meta, mask))
                    for name in METHODS:
                        value = arrays[f"{root}|instruction_{cond}|{name}|mse"].astype(float)
                        if len(value) != len(mask) or not np.isfinite(value).all():
                            raise ValueError("invalid saved errors")
                        if group == "all":
                            hrun = next(r for r in history["runs"] if r["split"] == split and r["seed"] == seed)
                            expected = hrun["sources"]["instruction"]["conditions"][cond]["populations"]["all"][name]["mse"]
                            if abs(value.mean() - expected) > 1e-6:
                                raise ValueError("saved error mean differs from source JSON")
                            hits.append(hrun["sources"]["instruction"]["conditions"][cond]["populations"]["all"]["_top1_is_gold_unit"]) if name == NORMAL else None
                        mets[name].append(value[mask])
                cell = {"methods": {name: summarize_seed_arrays(v, metas, args.bootstrap, 101) for name, v in mets.items()},
                        "memory_minus_normal": summarize_seed_arrays([a-b for a,b in zip(mets[MEMORY], mets[NORMAL])], metas, args.bootstrap, 109),
                        "strict_top1_mean": float(np.mean(hits)) if hits else None}
                payload["cohorts"][key] = cell
                for seed in (0, 1, 2):
                    for task in np.unique(metas[seed]["task"]):
                        mask = metas[seed]["task"] == task
                        for name in METHODS:
                            metric_rows.append({"cohort": key, "seed": seed, "task": int(task), "method": name,
                                                "rows": int(mask.sum()), "mse": float(mets[name][seed][mask].mean())})
                print("summarized", key, flush=True)
        for cond in ("D1", "D3", "D2"):
            agg = history["aggregate"][f"{split}|instruction|{cond}|all"]
            payload["fixed_budget_contents"][f"{split}_{cond}"] = {
                "unit": agg["N:unit:fix0.4"]["mse"]["mean"],
                "raw": agg["N:raw:fix0.4"]["mse"]["mean"],
                "unit_seed_sd": agg["N:unit:fix0.4"]["mse"]["sd"],
                "raw_seed_sd": agg["N:raw:fix0.4"]["mse"]["sd"]}
        route_errors, route_choices, route_folds, confidence = {}, {}, [], []
        for seed in (0, 1, 2):
            root = f"{split}_s{seed}|instruction_D3"
            a, b = (arrays[f"{root}|{name}|mse"].astype(float) for name in (NORMAL, MEMORY))
            sim, ent = (arrays[f"{root}|conf_{name}"].astype(float) for name in ("sim", "ent"))
            features = np.column_stack([sim, ent, sim**2, ent**2, sim*ent])
            choices, errors, folds = evaluate_gate(features, new_meta["task"], new_meta["episode"], a, b)
            for name, value in errors.items():
                route_errors.setdefault(name, []).append(value)
                prediction_arrays[f"{split}_s{seed}|{name}|mse"] = value
            for name, value in choices.items():
                route_choices.setdefault(name, []).append(value)
                prediction_arrays[f"{split}_s{seed}|{name}|choose_memory"] = value
            for k, value in new_meta.items():
                prediction_arrays[f"{split}_s{seed}|{k}"] = value
            confidence.append({"seed": seed, "rho_sim_vs_memory_minus_normal": float(spearmanr(sim, b-a).statistic),
                               "rho_entropy_conf_vs_memory_minus_normal": float(spearmanr(ent, b-a).statistic)})
            route_folds.extend([{**f, "seed": seed} for f in folds])
        routing = {"methods": {}, "folds": route_folds, "confidence": confidence}
        for name, errors in route_errors.items():
            cell = summarize_seed_arrays(errors, [new_meta]*3, args.bootstrap, 113)
            for ref in ("normal", "memory", "constant"):
                cell[f"minus_{ref}"] = summarize_seed_arrays([x-y for x,y in zip(errors, route_errors[ref])], [new_meta]*3, args.bootstrap, 127)
            if name in route_choices:
                cell["memory_choice_rate"] = float(np.mean([c.mean() for c in route_choices[name]]))
            elif name in ("normal", "memory", "random50_expected"):
                cell["memory_choice_rate"] = {"normal": 0., "memory": 1., "random50_expected": .5}[name]
            elif name == "oracle":
                cell["memory_choice_rate"] = float(np.mean([np.mean(b<a) for a,b in zip(route_errors["normal"],route_errors["memory"])]))
            routing["methods"][name] = cell
        payload["routing"][split] = routing
        print("routing complete", split, flush=True)
    payload["elapsed_seconds"] = time.monotonic() - started
    payload["source_unchanged"] = all(sha256(args.source / n) == h for n,h in payload["source_hashes"].items())
    if not payload["source_unchanged"]:
        raise ValueError("source changed during analysis")
    np.savez_compressed(args.output / "routing_predictions.npz", **prediction_arrays)
    with (args.output / "task_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metric_rows[0])); w.writeheader(); w.writerows(metric_rows)
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    write_report(args.output, payload)
    print("finished", args.output, "seconds", round(payload["elapsed_seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
