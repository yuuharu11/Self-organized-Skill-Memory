"""Measure cross-task retrieval ambiguity on held-out LIBERO-goal episodes.

The Split B towers are refit with a deterministic episode holdout inside the
goal suite. Training and memory contain only the remaining episodes; the
query set consists only of held-out episodes from goal task indices 10--19.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .abstracted_m1 import (
    _decode_residual_draws, _fit_abstracted_regressor, _mse,
    _predict_abstracted, _prepare_m1_train,
)
from .clip_retrieval import (
    CLIP_BATCH, CLIP_EPOCHS, CLIP_PATIENCE, DEFAULT_ANNOTATIONS,
    DEFAULT_PALIGEMMA_TOKENIZER, DEFAULT_PI05_BASE_PARAMS,
    _query_embedding, _suite_horizon, attach_subtask_texts,
    fit_dual_encoder, frozen_pi05_plan_embeddings, retrieve_embedding_top1,
)
from .compare_splits import _subset
from .experiment import (
    DEFAULT_CACHE, DEFAULT_DATASET, DEFAULT_OUTPUT_ROOT, load_rows,
)

GOAL_TASKS = tuple(range(10, 20))
SPLIT_B_TASKS = tuple(range(10, 30))
HORIZON = 25
HOLDOUT_FRACTION = 0.20
EARLY_INTERVALS = 3
III_T0 = 0.2


def split_goal_episodes(rows, *, seed=0, holdout_fraction=HOLDOUT_FRACTION):
    """Hold out goal episodes per task; never expose them to fit or memory."""
    goal_mask = np.isin(rows["task_index"], GOAL_TASKS)
    goal_episode_task = {}
    for episode, task in zip(rows["episode_id"][goal_mask], rows["task_index"][goal_mask]):
        previous = goal_episode_task.setdefault(str(episode), int(task))
        if previous != int(task):
            raise ValueError(f"episode {episode} maps to more than one task")
    rng = np.random.default_rng(seed)
    heldout = set()
    for task in GOAL_TASKS:
        episodes = np.asarray(sorted(ep for ep, value in goal_episode_task.items() if value == task))
        if len(episodes) < 2:
            raise ValueError(f"goal task {task} has fewer than two episodes")
        count = max(1, int(round(len(episodes) * holdout_fraction)))
        count = min(count, len(episodes) - 1)
        heldout.update(rng.choice(episodes, size=count, replace=False).tolist())
    query_mask = goal_mask & np.isin(rows["episode_id"], list(heldout))
    train_suite_mask = np.isin(rows["task_index"], SPLIT_B_TASKS) & ~np.isin(
        rows["episode_id"], list(heldout))
    train = _subset(rows, train_suite_mask)
    query = _subset(rows, query_mask)
    if not len(train["a"]) or not len(query["a"]):
        raise ValueError("episode split produced an empty training or query set")
    if np.intersect1d(train["episode_id"], query["episode_id"]).size:
        raise AssertionError("held-out query episodes leaked into training")
    if not np.isin(query["task_index"], GOAL_TASKS).all():
        raise AssertionError("query set must contain only LIBERO-goal tasks")
    return train, query, sorted(heldout)


def first_intervals_mask(rows, count=EARLY_INTERVALS):
    """Select each episode's first `count` rows in cached frame order."""
    result = np.zeros(len(rows["episode_id"]), dtype=bool)
    by_episode = {}
    for index, episode in enumerate(rows["episode_id"]):
        by_episode.setdefault(str(episode), []).append(index)
    for indices in by_episode.values():
        indices.sort(key=lambda i: int(rows["frame"][i]))
        result[indices[:count]] = True
    return result


def retrieval_metrics(query_rows, memory_rows, gold_b, memory_b, chosen, predictions,
                      actions, *, early_mask, t0=III_T0, seed=20261027):
    """Summarize task/phrase identity, base distance, and (iii) MSE."""
    chosen = np.asarray(chosen, np.int64)
    retrieved_task = memory_rows["task_index"][chosen]
    same_task = retrieved_task == query_rows["task_index"]
    query_phrase = np.asarray(query_rows["subtask_text"], dtype=str)
    memory_phrase = np.asarray(memory_rows["subtask_text"], dtype=str)[chosen]
    phrase_valid = query_phrase != ""
    same_phrase = memory_phrase == query_phrase

    # Predictions were generated with the exact same per-row M1 and Fiii
    # decoding seeds for every retrieval condition.
    b_distance = np.linalg.norm(np.asarray(memory_b)[chosen] - gold_b, axis=1)
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim != actions.ndim + 1 or predictions.shape[1:] != actions.shape:
        raise ValueError(f"(iii) predictions must be draws×actions, got {predictions.shape} vs {actions.shape}")
    squared_error = (predictions - actions[None]) ** 2
    action_mse_rows = squared_error.mean(axis=(0, *range(2, squared_error.ndim)))

    def aggregate(mask):
        phrase_mask = mask & phrase_valid
        return {
            "n": int(mask.sum()),
            "same_task_rate": float(same_task[mask].mean()) if mask.any() else None,
            "same_phrase_rate": (float(same_phrase[phrase_mask].mean())
                                 if phrase_mask.any() else None),
            "phrase_n": int(phrase_mask.sum()),
            "retrieved_gold_b_l2": float(b_distance[mask].mean()) if mask.any() else None,
            "iii_action_mse": float(action_mse_rows[mask].mean()) if mask.any() else None,
        }

    return {"all": aggregate(np.ones(len(chosen), bool)), "early": aggregate(early_mask)}


def _mean_embeddings(texts, *, params_path, tokenizer_path):
    unique, inverse = np.unique(np.asarray(texts, dtype=str), return_inverse=True)
    vectors = frozen_pi05_plan_embeddings(
        unique, params_path=params_path, tokenizer_path=tokenizer_path)
    return vectors[inverse]


def _memory_query_embedding(model, fit, rows, bases, *, device, include_img, include_state):
    q = _query_embedding(model, fit, rows, device=device,
                         include_img=include_img, include_state=include_state)
    with torch.no_grad():
        h = model.encode_skill(torch.as_tensor(bases, dtype=torch.float32,
                                               device=device)).cpu().numpy()
    return q, h


def _report(results):
    lines = [
        "# Same-scene, different-instruction retrieval diagnosis",
        "",
        "Split B episode holdout: the query tower is trained on object+goal training episodes, while held-out episodes from all ten goal tasks are excluded from both fitting and memory. Memory is the goal-task slice of those training episodes, so a uniform task baseline is 10%. The five query inputs are compared with asymmetric g(query)·h(b) and symmetric g(query)·g(memory) retrieval.",
        "",
        f"The early subset is the first {EARLY_INTERVALS} cached H=25-valid intervals per held-out episode, ordered by frame. `same phrase` is computed only over queries with a non-empty subtask plan phrase. (iii) action MSE uses t0={III_T0} and paired residual/noise seeds across retrieval conditions.",
        "",
        "| Query input | Retrieval | N | Same task | Same phrase (N) | Retrieved b ↔ gold b L2 | (iii) action MSE | Early N | Early same task | Early same phrase (N) | Early b L2 | Early (iii) MSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["rows"]:
        all_m, early = row["all"], row["early"]
        def fmt(value):
            return "—" if value is None else f"{value:.4f}"
        lines.append(
            f"| {row['query_input']} | {row['retrieval']} | {all_m['n']} | {fmt(all_m['same_task_rate'])} | {fmt(all_m['same_phrase_rate'])} ({all_m['phrase_n']}) | {fmt(all_m['retrieved_gold_b_l2'])} | {fmt(all_m['iii_action_mse'])} | {early['n']} | {fmt(early['same_task_rate'])} | {fmt(early['same_phrase_rate'])} ({early['phrase_n']}) | {fmt(early['retrieved_gold_b_l2'])} | {fmt(early['iii_action_mse'])} |"
        )
    lines.extend([
        "",
        f"Empirical random same-task baseline from the goal-only memory task mix: {results['random_task_baseline']:.4f}; uniform ten-task baseline: 0.1000.",
        "",
        "## Notes",
        "",
        "Task identity is the LeRobot task_index. Subtask phrase identity is exact string equality after joining eval_skill_segment to the annotated plan; segment -1 maps to the empty string. Instructions and plan phrases use frozen local π0.5 PaliGemma token embeddings averaged over SentencePiece tokens. No weights are downloaded.",
        "",
    ])
    return "\n".join(lines)


def run(*, output_root=DEFAULT_OUTPUT_ROOT, cache_dir=DEFAULT_CACHE,
        dataset_dir=DEFAULT_DATASET, annotations_path=DEFAULT_ANNOTATIONS,
        params_path=DEFAULT_PI05_BASE_PARAMS,
        tokenizer_path=DEFAULT_PALIGEMMA_TOKENIZER,
        seed=0, device_name="cuda"):
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device(device_name)
    rows = load_rows(cache_dir, dataset_dir, max_horizon=HORIZON)
    rows = attach_subtask_texts(rows, cache_dir=cache_dir,
                                annotations_path=annotations_path)
    manifest = json.loads((Path(cache_dir) / "manifest.json").read_text())
    task_for_episode = {str(item["episode_id"]): str(item["task"])
                        for item in manifest["episodes"]}
    rows["task_instruction"] = np.asarray(
        [task_for_episode[str(ep)] for ep in rows["episode_id"]], dtype=str)
    if len(np.unique(rows["task_instruction"][np.isin(rows["task_index"], GOAL_TASKS)])) != 10:
        raise AssertionError("expected ten distinct goal instructions")

    plan_vectors = _mean_embeddings(rows["subtask_text"], params_path=params_path,
                                    tokenizer_path=tokenizer_path)
    instruction_vectors = _mean_embeddings(rows["task_instruction"], params_path=params_path,
                                           tokenizer_path=tokenizer_path)
    rows["subtask_embedding"] = plan_vectors
    train, query, heldout_episodes = split_goal_episodes(rows, seed=seed)
    # Use only the ten goal-task portion as memory to directly test cross-task
    # ambiguity; all memory episodes are still from the Split B training set.
    memory_rows = _suite_horizon(train, GOAL_TASKS, HORIZON)
    train = _suite_horizon(train, SPLIT_B_TASKS, HORIZON)
    query = _suite_horizon(query, GOAL_TASKS, HORIZON)
    if not len(query["a"]) or not len(memory_rows["a"]):
        raise ValueError("no H=25-valid held-out query or goal memory rows")
    if np.intersect1d(query["episode_id"], memory_rows["episode_id"]).size:
        raise AssertionError("held-out goal episodes appear in retrieval memory")

    train_actions = train["a"][:, :HORIZON].copy()
    query_actions = query["a"][:, :HORIZON].copy()
    m1 = _prepare_m1_train(train, train_actions, seed=seed + HORIZON,
                           epochs=60, batch_size=512, device=device)
    train_b, _ = m1.encode(train_actions, train["s"])
    memory_b, _ = m1.encode(memory_rows["a"][:, :HORIZON], memory_rows["s"])
    gold_b, _ = m1.encode(query_actions, query["s"])
    train_mids = _decode_residual_draws(m1, train_b, train["f"], train["s"],
                                        draws=4, seed=seed + HORIZON * 1000 + 11)
    fiii, fiii_fit = _fit_abstracted_regressor(
        m1, train, train_actions, train_b, train_mids, epochs=150,
        batch_size=512, patience=8, seed=seed + HORIZON * 100 + 3,
        device=device)
    # Keep held-out gold action only for evaluation; no query row enters fit.
    early_mask = first_intervals_mask(query)

    # Different E_sub inputs share the exact same local frozen PaliGemma
    # embedding construction. The instruction-only model suppresses image and
    # state channels both while fitting and while making query/memory keys.
    variants = [
        ("image + state", "none", None, True, True, 0),
        ("+ verb class", "verb_class", None, True, True, 1),
        ("+ subtask phrase", "sentence", "subtask_embedding", True, True, 2),
        ("+ instruction sentence", "sentence", "instruction_embedding", True, True, 3),
        ("instruction only", "sentence", "instruction_embedding", False, False, 3),
    ]
    # Rows have been filtered/reordered, so attach vectors by episode+frame key.
    vector_lookup = {(str(ep), int(frame)): instruction_vectors[i]
                     for i, (ep, frame) in enumerate(zip(rows["episode_id"], rows["frame"]))}
    for subset in (train, memory_rows, query):
        subset["instruction_embedding"] = np.stack([
            vector_lookup[(str(ep), int(frame))]
            for ep, frame in zip(subset["episode_id"], subset["frame"])])

    all_results = []
    goal_memory_task_counts = np.asarray([
        np.sum(memory_rows["task_index"] == task) for task in GOAL_TASKS], dtype=np.float64)
    random_task_baseline = float(np.sum((goal_memory_task_counts / goal_memory_task_counts.sum()) ** 2))

    for label, subtask_mode, input_key, include_img, include_state, mode_seed in variants:
        if input_key:
            train["subtask_embedding"] = train[input_key]
            memory_rows["subtask_embedding"] = memory_rows[input_key]
            query["subtask_embedding"] = query[input_key]
        model, fit = fit_dual_encoder(
            train, train_b, use_subtask=subtask_mode != "none",
            subtask_mode=subtask_mode, nearby_positive=False,
            seed=seed + HORIZON * 17 + 1000 * mode_seed,
            epochs=CLIP_EPOCHS, patience=CLIP_PATIENCE, device=device,
            allowed_task_indices=SPLIT_B_TASKS,
            include_img=include_img, include_state=include_state)

        memory_query = _query_embedding(model, fit, memory_rows, device=device,
                                        include_img=include_img,
                                        include_state=include_state)
        query_embedding = _query_embedding(model, fit, query, device=device,
                                           include_img=include_img,
                                           include_state=include_state)
        with torch.no_grad():
            memory_skill = model.encode_skill(torch.as_tensor(
                memory_b, dtype=torch.float32, device=device)).cpu().numpy()
        for retrieval_name, memory_embedding in (("asymmetric g·h", memory_skill),
                                                  ("symmetric g·g", memory_query)):
            chosen = retrieve_embedding_top1(query_embedding, memory_embedding)
            retrieved_b = memory_b[chosen]
            predictions = _predict_abstracted(
                m1, fiii, retrieved_b, query["f"], query["s"],
                t0=III_T0, seed=seed + 2500 + HORIZON * 100,
                draws=4)
            metrics = retrieval_metrics(
                query, memory_rows, gold_b, memory_b, chosen, predictions,
                query_actions, early_mask=early_mask)
            all_results.append({"query_input": label, "retrieval": retrieval_name,
                                **metrics["all"], "all": metrics["all"],
                                "early": metrics["early"],
                                "epochs": fit["epochs"],
                                "best_validation_loss": fit["best_validation_loss"],
                                "modality_weights": model.modality_weights()})

    results = {
        "seed": seed, "split": "B: object+goal fit, held-out goal episodes queried",
        "horizon": HORIZON, "early_intervals_per_episode": EARLY_INTERVALS,
        "holdout_fraction_per_goal_task": HOLDOUT_FRACTION,
        "heldout_goal_episode_count": len(heldout_episodes),
        "train_rows": len(train["a"]), "goal_memory_rows": len(memory_rows["a"]),
        "query_rows": len(query["a"]), "m1": m1.describe(),
        "fiii_best_epoch": fiii_fit["best_epoch"],
        "fiii_best_validation_loss": fiii_fit["best_validation_loss"],
        "random_task_baseline": random_task_baseline,
        "query_task_counts": {str(t): int(np.sum(query["task_index"] == t)) for t in GOAL_TASKS},
        "memory_task_counts": {str(t): int(np.sum(memory_rows["task_index"] == t)) for t in GOAL_TASKS},
        "rows": all_results,
    }
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / (datetime.now(timezone.utc).strftime("task-ambiguity-%Y%m%dT%H%M%SZ")
                            + "-" + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    (output / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (output / "report.md").write_text(_report(results))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--pi05-base-params", type=Path, default=DEFAULT_PI05_BASE_PARAMS)
    parser.add_argument("--paligemma-tokenizer", type=Path, default=DEFAULT_PALIGEMMA_TOKENIZER)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    output = run(output_root=args.output_root, cache_dir=args.cache_dir,
                 dataset_dir=args.dataset, annotations_path=args.annotations,
                 params_path=args.pi05_base_params, tokenizer_path=args.paligemma_tokenizer,
                 seed=args.seed, device_name=args.device)
    print(f"output: {output}")


if __name__ == "__main__":
    main()
