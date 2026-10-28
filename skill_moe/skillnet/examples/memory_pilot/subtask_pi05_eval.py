"""Run frame-aligned subtask generation/metrics for the plain pi05 model.

The feature cache is sampled at stride five. This evaluator predicts only those
existing cache rows so outputs can be joined by (episode_id, step) without
changing the cache or the source LeRobot dataset.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import pathlib
import re
from collections import defaultdict

import jax
import numpy as np
from flax import nnx
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies import libero_policy
from openpi.shared import nnx_utils
from openpi.training import config_moe_skill as _config
from openpi.training import data_loader_skill as _loader
from openpi.training import weight_loaders as _weight_loaders
from openpi import transforms as _transforms


def _task_groups(root: pathlib.Path) -> tuple[dict[int, str], dict[int, list[int]], dict[int, int]]:
    tasks = {int(row["task_index"]): row["task"] for row in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines()) if row}
    episodes_by_task: dict[int, list[int]] = defaultdict(list)
    episode_lengths: dict[int, int] = {}
    for line in (root / "meta/episodes.jsonl").read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        episode = int(row["episode_index"])
        episode_lengths[episode] = int(row["length"])
        for task in row["tasks"]:
            for task_index, name in tasks.items():
                if name == task:
                    episodes_by_task[task_index].append(episode)
                    break
    return tasks, episodes_by_task, episode_lengths


def _heldout_episode_ids(task_indices: tuple[int, ...], episodes_by_task: dict[int, list[int]], fraction: float) -> set[int]:
    heldout: set[int] = set()
    for task_index in task_indices:
        episodes = sorted(set(episodes_by_task.get(task_index, [])))
        n = int(round(len(episodes) * fraction))
        if fraction and n == 0 and len(episodes) > 1:
            n = 1
        if n:
            heldout.update(episodes[-n:])
    return heldout


def _verb(text: str) -> str:
    text = re.sub(r"[^a-z ]", " ", text.lower())
    words = text.split()
    for phrase in ("pick up", "turn on", "turn off", "put away", "pick", "place", "open", "close", "push", "turn"):
        if text.startswith(phrase + " ") or text == phrase:
            return "pick" if phrase == "pick up" else phrase
    return words[0] if words else ""


def _object(text: str) -> str:
    text = re.sub(r"[^a-z ]", " ", text.lower()).strip()
    text = re.sub(r"^(pick up|turn on|turn off|put away|pick|place|open|close|push|turn)\s+", "", text)
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.split(r"\b(?:next to|between|under|over|beside|onto|inside|from|on|in|into|to|at)\b", text)[0]
    return re.sub(r"\s+", " ", text).strip()


def _verb_match(gold: str, pred: str) -> bool:
    return bool(_verb(gold)) and _verb(gold) == _verb(pred)


def _object_match(gold: str, pred: str) -> bool:
    gold_obj, pred_obj = _object(gold), _object(pred)
    return bool(gold_obj) and (gold_obj == pred_obj or gold_obj in pred_obj or pred_obj in gold_obj)


def _metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "exact_match": None, "verb_match": None, "object_match": None}
    return {
        "n": len(rows),
        "exact_match": float(np.mean([r["gold"].strip().lower() == r["prediction"].strip().lower() for r in rows])),
        "verb_match": float(np.mean([_verb_match(r["gold"], r["prediction"]) for r in rows])),
        "object_match": float(np.mean([_object_match(r["gold"], r["prediction"]) for r in rows])),
    }


def _boundary_metrics(rows: list[dict]) -> dict:
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)
    errors: list[int] = []
    misses = 0
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda r: r["step"])
        gt_changes = [i for i in range(1, len(episode_rows)) if episode_rows[i]["gold"] != episode_rows[i - 1]["gold"]]
        pred_changes = [i for i in range(1, len(episode_rows)) if episode_rows[i]["prediction"] != episode_rows[i - 1]["prediction"]]
        unused = set(pred_changes)
        for gt_i in gt_changes:
            if not unused:
                misses += 1
                continue
            pred_i = min(unused, key=lambda i: abs(episode_rows[i]["step"] - episode_rows[gt_i]["step"]))
            unused.remove(pred_i)
            errors.append(int(episode_rows[pred_i]["step"] - episode_rows[gt_i]["step"]))
    return {
        "n_boundaries": len(errors) + misses,
        "matched": len(errors),
        "missed": misses,
        "mean_signed_step_error": float(np.mean(errors)) if errors else None,
        "mean_absolute_step_error": float(np.mean(np.abs(errors))) if errors else None,
        "early": sum(x < 0 for x in errors),
        "late": sum(x > 0 for x in errors),
        "on_time": sum(x == 0 for x in errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("A", "B"), required=True)
    parser.add_argument("--checkpoint", required=True, help="params directory or pi05_base")
    parser.add_argument("--cache-manifest", default="/work/.skillnet-state/outputs/memory-pilot/emergent-light-20261026T050453Z-0cb536/feature-cache/manifest.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Debug/sample count; 0 evaluates every selected cache row")
    parser.add_argument("--training-suite-all", action="store_true",
                        help="Predict every cache row in this split's training suites, rather than only the held-out 20%%")
    parser.add_argument("--training-suite-missing-only", action="store_true",
                        help="Predict the training-suite episodes absent from the existing 20%% held-out cache")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="Resume from predictions.partial.jsonl in an existing output directory")
    args = parser.parse_args()
    if args.training_suite_all and args.training_suite_missing_only:
        parser.error("choose at most one of --training-suite-all and --training-suite-missing-only")

    out = pathlib.Path(args.output)
    if args.resume:
        if not out.is_dir():
            raise FileNotFoundError(f"Cannot resume; output directory does not exist: {out}")
    else:
        out.mkdir(parents=True, exist_ok=False)
    cache_manifest = json.loads(pathlib.Path(args.cache_manifest).read_text())
    root = pathlib.Path(cache_manifest["dataset"])
    annotations_file = pathlib.Path(__file__).resolve().parents[1] / "libero/annotations/instruct2plan_40.json"
    annotations = {k: v["plan"] for k, v in json.loads(annotations_file.read_text()).items()}
    train_tasks = tuple(range(20, 30)) + (tuple(range(30, 40)) if args.split == "A" else tuple(range(10, 20)))
    unseen_tasks = tuple(range(10, 20)) if args.split == "A" else tuple(range(30, 40))
    tasks, episodes_by_task, lengths = _task_groups(root)
    heldout = _heldout_episode_ids(train_tasks, episodes_by_task, 0.2)
    if args.training_suite_all or args.training_suite_missing_only:
        all_training_episodes = {episode for task_index in train_tasks
                                 for episode in episodes_by_task.get(task_index, [])}
        eval_episodes = (all_training_episodes if args.training_suite_all else
                         all_training_episodes - heldout)
        selected_tasks = train_tasks
    else:
        eval_episodes = set(heldout)
        for task_index in unseen_tasks:
            eval_episodes.update(episodes_by_task.get(task_index, []))
        selected_tasks = (*train_tasks, *unseen_tasks)

    records = []
    for episode_info in cache_manifest["episodes"]:
        episode_id = episode_info["episode_id"]
        episode_num = int(episode_id.rsplit("-", 1)[-1])
        task_index = int(episode_info["task_index"])
        if episode_num not in eval_episodes or task_index not in selected_tasks:
            continue
        group = ("train_suite_all" if args.training_suite_all else
                 "train_suite_missing" if args.training_suite_missing_only else
            "train_suite_heldout" if task_index in train_tasks else "unseen_suite")
        with np.load(episode_info["cache"]) as cache:
            for step in cache["step"].astype(int).tolist():
                records.append({"episode_id": episode_id, "episode_num": episode_num, "task_index": task_index, "step": step, "group": group})
    records.sort(key=lambda r: (r["episode_num"], r["step"]))
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise ValueError("No matching cache rows found for the selected split")

    partial_path = out / "predictions.partial.jsonl"
    predictions = []
    if args.resume:
        if not partial_path.exists():
            raise FileNotFoundError(f"Cannot resume; missing partial predictions file: {partial_path}")
        with partial_path.open() as f:
            predictions = [json.loads(line) for line in f if line.strip()]
    done_keys = {(row["episode_id"], int(row["step"])) for row in predictions}
    records = [row for row in records if (row["episode_id"], int(row["step"])) not in done_keys]

    selected_episodes = sorted({r["episode_num"] for r in records})
    starts = {}
    offset = 0
    for episode_num in selected_episodes:
        starts[episode_num] = offset
        offset += lengths[episode_num]
    config = _config.get_config(f"pi05_libero40_subtask_split_{args.split.lower()}")
    data_cfg = config.data.create(pathlib.Path("/tmp"), config.model)
    meta = LeRobotDatasetMetadata(config.data.repo_id, root=str(root))
    dataset = LeRobotDataset(
        config.data.repo_id,
        root=str(root),
        episodes=selected_episodes,
        delta_timestamps={key: [t / meta.fps for t in range(config.model.action_horizon)] for key in data_cfg.action_sequence_keys},
    )
    episode_to_local = {episode: i for i, episode in enumerate(selected_episodes)}
    original_get_indices = dataset._get_query_indices
    dataset._get_query_indices = lambda row, episode: original_get_indices(row, episode_to_local[episode])
    dataset = _loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset.meta.tasks)])
    infer_model_transforms = _transforms.Group(inputs=[
        _transforms.InjectDefaultPrompt(None),
        _transforms.ResizeImages(224, 224),
        _transforms.AttachLiberoSubtask(annotations),
        _transforms.TokenizeSubtaskInference(PaligemmaTokenizer(config.model.max_token_len)),
        _transforms.PadStatesAndActions(config.model.action_dim),
    ])
    eval_cfg = dataclasses.replace(data_cfg, model_transforms=infer_model_transforms, norm_stats={})
    transformed = _loader.transform_dataset(dataset, eval_cfg, skip_norm_stats=True)

    model = config.model.create(jax.random.key(0))
    state = nnx.state(model)
    loaded = _weight_loaders.CheckpointWeightLoader(args.checkpoint).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    nnx.update(model, state)
    generate = nnx_utils.module_jit(model.generate_subtask, static_argnames=("max_tokens",))
    tokenizer = PaligemmaTokenizer(config.model.max_token_len)

    policy_route = {"verified": False}
    bs = args.batch_size
    partial_file = partial_path.open("a", buffering=1)
    for begin in range(0, len(records), bs):
        batch_records = records[begin:begin + bs]
        samples = []
        for row in batch_records:
            local_ep = episode_to_local[row["episode_num"]]
            raw_start = int(np.asarray(dataset._dataset.episode_data_index["from"])[local_ep])
            sample = transformed[raw_start + row["step"]]
            row["gold"] = str(sample["subtask"])
            row["prompt"] = tasks[row["task_index"]]
            samples.append(sample)
        batch = jax.tree.map(lambda *xs: np.stack(xs), *[
            {k: v for k, v in s.items() if k in {"image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask"}}
            for s in samples
        ])
        # Policy.infer converts transformed NumPy leaves to JAX arrays before
        # calling _maybe_generate_subtask; mirror that production path here.
        batch = jax.tree.map(jax.numpy.asarray, batch)
        observation = _model.Observation.from_dict(batch)
        token_ids = np.asarray(generate(observation, max_tokens=args.max_tokens))
        if not policy_route["verified"]:
            from openpi.policies.policy import Policy

            os.environ["SKILLNET_SUBTASK_MAX_TOKENS"] = str(args.max_tokens)
            policy = Policy(model)
            policy_observation = policy._maybe_generate_subtask(observation, prompt=batch_records[0]["prompt"])
            if policy_observation.tokenized_prompt.shape[-1] != config.model.max_token_len + args.max_tokens:
                raise AssertionError("Policy subtask route did not attach the generated tokens")
            policy_route = {
                "verified": True,
                "generated_shape": list(policy_observation.tokenized_prompt.shape),
                "max_tokens": args.max_tokens,
            }
        for row, tokens in zip(batch_records, token_ids, strict=True):
            prediction = {**row, "prediction": tokenizer.detokenize(tokens)}
            predictions.append(prediction)
            partial_file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        partial_file.flush()
    partial_file.close()

    groups = (("train_suite_all",) if args.training_suite_all else
              ("train_suite_missing",) if args.training_suite_missing_only else
              ("train_suite_heldout", "unseen_suite"))
    by_group = {group: _metrics([r for r in predictions if r["group"] == group]) for group in groups}
    report = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "cache_manifest": args.cache_manifest,
        "cache_stride_frames": cache_manifest.get("stride"),
        "max_tokens": args.max_tokens,
        "rows": len(predictions),
        "groups": by_group,
        "boundaries": _boundary_metrics(predictions),
        "policy_route": policy_route,
        "notes": {
            "split_task_ids": list(train_tasks),
            "unseen_task_ids": list(unseen_tasks),
            "heldout_episode_fraction": 0.2,
            "heldout_selection": "last sorted episode IDs per task; same selection used by training loader",
            "training_suite_all": bool(args.training_suite_all),
            "training_suite_missing_only": bool(args.training_suite_missing_only),
            "cache_join_key": ["episode_id", "step"],
            "evaluation_frames": "all selected rows in the existing feature cache; one row every cache stride",
            "feature_cache_suite_labels_ignored": True,
            "no_segment_label": "none",
        },
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "predictions.jsonl").open("w") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "cache_aligned_subtasks.jsonl").open("w") as f:
        for row in predictions:
            f.write(json.dumps({"episode_id": row["episode_id"], "step": row["step"], "subtask": row["prediction"]}, ensure_ascii=False) + "\n")
    with (out / "examples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("group", "episode_id", "step", "prompt", "gold", "prediction"), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predictions[:10])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
