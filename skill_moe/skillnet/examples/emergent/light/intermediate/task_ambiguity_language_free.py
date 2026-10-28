"""Repeat the held-out LIBERO-goal ambiguity study with language-free keys."""
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
    CLIP_EPOCHS, CLIP_PATIENCE, DEFAULT_ANNOTATIONS,
    DEFAULT_PALIGEMMA_TOKENIZER, DEFAULT_PI05_BASE_PARAMS,
    _query_embedding, attach_subtask_texts, fit_dual_encoder,
    frozen_pi05_plan_embeddings, retrieve_embedding_top1,
)
from .compare_splits import _subset
from .experiment import (
    DEFAULT_CACHE, DEFAULT_DATASET, DEFAULT_OUTPUT_ROOT,
    _episode_train_validation_indices, _standardize, _standardize_fit,
    load_rows,
)
from .task_ambiguity import (
    EARLY_INTERVALS, GOAL_TASKS, SPLIT_B_TASKS,
    first_intervals_mask, retrieval_metrics, split_goal_episodes,
)

DEFAULT_CAMERA_CACHE = Path(
    "/work/.skillnet-state/outputs/memory-pilot/"
    "emergent-light-keys-20261026T074213Z-9e39c3/camera_features")
HORIZON = 25
III_T0 = 0.2


def attach_wrist_features(rows, camera_cache):
    """Join cached language-free wrist-camera vectors by episode and frame."""
    camera_cache = Path(camera_cache)
    manifest = json.loads((camera_cache / "manifest.json").read_text())
    by_episode = {str(item["episode_id"]): item for item in manifest["episodes"]}
    result = np.empty((len(rows["episode_id"]), 2048), dtype=np.float32)
    cache_data = {}
    for index, (episode, frame) in enumerate(zip(rows["episode_id"], rows["frame"])):
        episode = str(episode)
        if episode not in by_episode:
            raise KeyError(f"episode {episode} is missing from wrist camera cache")
        if episode not in cache_data:
            path = camera_cache / "episodes" / f"{episode}.npz"
            with np.load(path, allow_pickle=False) as data:
                steps = np.asarray(data["step"], dtype=np.int64)
                wrist = np.asarray(data["wrist"], dtype=np.float32)
            if wrist.ndim != 2 or wrist.shape[1] != result.shape[1]:
                raise ValueError(f"unexpected wrist feature shape {wrist.shape} for {episode}")
            cache_data[episode] = {int(step): wrist[i] for i, step in enumerate(steps)}
        try:
            result[index] = cache_data[episode][int(frame)]
        except KeyError as exc:
            raise KeyError(f"frame {frame} is absent from wrist feature cache for {episode}") from exc
    if not np.isfinite(result).all():
        raise ValueError("wrist feature cache contains non-finite vectors")
    return result


def _initial_rows(rows, episodes):
    mask = np.isin(rows["episode_id"], np.asarray(list(episodes), dtype=str))
    subset = _subset(rows, mask)
    selected = []
    for episode in np.unique(subset["episode_id"]):
        ix = np.flatnonzero(subset["episode_id"] == episode)
        selected.append(ix[np.argmin(subset["frame"][ix])])
    return _subset(subset, np.isin(np.arange(len(subset["a"])), selected))


def initial_task_accuracy(train_rows, query_rows, feature_name):
    """Fit a regularized linear classifier on initial-frame features."""
    x_train = np.asarray(train_rows[feature_name], np.float32)
    y_train = np.asarray(train_rows["task_index"], np.int64)
    x_query = np.asarray(query_rows[feature_name], np.float32)
    y_query = np.asarray(query_rows["task_index"], np.int64)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x_train = ((x_train - mean) / scale).astype(np.float32)
    x_query = ((x_query - mean) / scale).astype(np.float32)
    classes = np.unique(y_train)
    lookup = {int(label): i for i, label in enumerate(classes)}
    y = np.asarray([lookup[int(label)] for label in y_train], np.int64)
    torch.manual_seed(0)
    classifier = torch.nn.Linear(x_train.shape[1], len(classes))
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.03, weight_decay=0.01)
    xt = torch.as_tensor(x_train)
    yt = torch.as_tensor(y)
    for _ in range(200):
        logits = classifier(xt)
        loss = torch.nn.functional.cross_entropy(logits, yt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = classes[classifier(torch.as_tensor(x_query)).argmax(dim=-1).numpy()]
    return {"accuracy": float(np.mean(prediction == y_query)),
            "query_count": int(len(y_query)),
            "train_count": int(len(y_train)),
            "macro_accuracy": float(np.mean([
                np.mean(prediction[y_query == task] == y_query[y_query == task])
                for task in np.unique(y_query)]))}


def _feature_variant(rows, feature_source):
    variant = {name: value.copy() if isinstance(value, np.ndarray) else value
               for name, value in rows.items()}
    if feature_source in ("original", "f+k (reference)"):
        return variant
    variant["f"] = np.zeros_like(variant["f"], dtype=np.float32)
    if feature_source in ("k", "k only"):
        return variant
    if feature_source in ("wrist", "wrist only"):
        variant["k"] = variant["wrist_feature"].astype(np.float32, copy=True)
        return variant
    raise ValueError(feature_source)


def _make_report(result):
    lines = [
        "# Language-free feature check for task ambiguity",
        "",
        "The `f+k` reference reproduces the previous image+state setting. All other image paths zero f and use only the language-free k vector or the independently cached language-free wrist-camera vector; state remains enabled except in the instruction-only row. The Split B tower fits on object+goal training episodes. Held-out goal episodes are excluded from both tower fitting and memory. The memory is limited to training episodes from goal tasks 10–19 to retain the 10% task chance baseline.",
        "",
        f"Early results use the first {EARLY_INTERVALS} H=25-valid intervals in each held-out episode. Subtask phrase equality is exact plan-text equality and is calculated only for non-empty query phrases. (iii) action MSE is at t0={III_T0}.",
        "",
        "| Image feature input | Query condition | Retrieval | N | Same task | Same phrase (N) | Retrieved b ↔ gold b L2 | (iii) action MSE | Early same task | Early same phrase (N) | Early b L2 | Early (iii) MSE | Initial-frame task accuracy (feature only) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        a, e = row["all"], row["early"]
        def fmt(v): return "—" if v is None else f"{v:.4f}"
        acc = result["initial_task_accuracy"].get(row["feature_source"])
        acc_text = fmt(acc["accuracy"]) if acc else "—"
        lines.append(
            f"| {row['feature_source']} | {row['query_input']} | {row['retrieval']} | {a['n']} | {fmt(a['same_task_rate'])} | {fmt(a['same_phrase_rate'])} ({a['phrase_n']}) | {fmt(a['retrieved_gold_b_l2'])} | {fmt(a['iii_action_mse'])} | {fmt(e['same_task_rate'])} | {fmt(e['same_phrase_rate'])} ({e['phrase_n']}) | {fmt(e['retrieved_gold_b_l2'])} | {fmt(e['iii_action_mse'])} | {acc_text} |"
        )
    lines.extend([
        "",
        f"Uniform task chance is 0.1000; empirical chance under the held-out-query / goal-memory task mix is {result['random_task_baseline']:.4f}.",
        "",
        "The initial-frame task classifier is a regularized linear classifier trained only on initial frames of goal-task memory episodes and evaluated on held-out query episodes. It receives one feature vector only (k, wrist, or f+k reference), no proprioception, language, action, or future frame. Wrist vectors came from the existing `camera_features` cache and were joined by episode and frame; no feature re-extraction was performed.",
        "",
    ])
    return "\n".join(lines)


def run(*, output_root=DEFAULT_OUTPUT_ROOT, cache_dir=DEFAULT_CACHE,
        dataset_dir=DEFAULT_DATASET, camera_cache=DEFAULT_CAMERA_CACHE,
        annotations_path=DEFAULT_ANNOTATIONS, params_path=DEFAULT_PI05_BASE_PARAMS,
        tokenizer_path=DEFAULT_PALIGEMMA_TOKENIZER, seed=0, device_name="cuda"):
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
    rows["wrist_feature"] = attach_wrist_features(rows, camera_cache)
    unique_texts, inverse = np.unique(rows["subtask_text"], return_inverse=True)
    rows["subtask_embedding"] = frozen_pi05_plan_embeddings(
        unique_texts, params_path=params_path,
        tokenizer_path=tokenizer_path)[inverse]
    unique_instructions, inverse = np.unique(rows["task_instruction"], return_inverse=True)
    rows["instruction_embedding"] = frozen_pi05_plan_embeddings(
        unique_instructions, params_path=params_path,
        tokenizer_path=tokenizer_path)[inverse]

    train_raw, query_raw, heldout = split_goal_episodes(rows, seed=seed)
    train = _subset(train_raw, train_raw["valid_h25"])
    query = _subset(query_raw, query_raw["valid_h25"])
    memory = _subset(train, np.isin(train["task_index"], GOAL_TASKS))
    if np.intersect1d(query["episode_id"], memory["episode_id"]).size:
        raise AssertionError("held-out query episode leaked into memory")
    if not np.isin(train["task_index"], SPLIT_B_TASKS).all():
        raise AssertionError("training rows include suites outside Split B")

    train_a = train["a"][:, :HORIZON].copy()
    query_a = query["a"][:, :HORIZON].copy()
    memory_a = memory["a"][:, :HORIZON].copy()
    m1 = _prepare_m1_train(train, train_a, seed=seed + HORIZON,
                           epochs=60, batch_size=512, device=device)
    train_b, _ = m1.encode(train_a, train["s"])
    memory_b, _ = m1.encode(memory_a, memory["s"])
    gold_b, _ = m1.encode(query_a, query["s"])
    train_mid = _decode_residual_draws(m1, train_b, train["f"], train["s"],
                                       draws=4, seed=seed + HORIZON * 1000 + 11)
    fiii, fiii_fit = _fit_abstracted_regressor(
        m1, train, train_a, train_b, train_mid, epochs=150, batch_size=512,
        patience=8, seed=seed + HORIZON * 100 + 3, device=device)
    early_mask = first_intervals_mask(query)
    residual_seed = seed + 5000
    query_eps = np.random.default_rng(residual_seed + 9001).standard_normal(
        (4, len(query_a), HORIZON * query_a.shape[-1])).astype(np.float32)

    memory_episode_ids = np.unique(memory["episode_id"])
    query_episode_ids = np.unique(query["episode_id"])
    initial_train_raw = _initial_rows(rows, memory_episode_ids)
    initial_query_raw = _initial_rows(rows, query_episode_ids)
    initial_accuracy = {}
    for label, feature in (("k only", "k"), ("wrist only", "wrist_feature"),
                          ("f+k (reference)", "f")):
        train_init = dict(initial_train_raw)
        query_init = dict(initial_query_raw)
        if label == "f+k (reference)":
            train_init["f"] = np.concatenate([train_init["f"], train_init["k"]], axis=-1)
            query_init["f"] = np.concatenate([query_init["f"], query_init["k"]], axis=-1)
            feature_name = "f"
        else:
            feature_name = feature
        initial_accuracy[label] = initial_task_accuracy(train_init, query_init, feature_name)

    feature_sources = (
        ("f+k (reference)", "original"), ("k only", "k"),
        ("wrist only", "wrist"),
    )
    conditions = (
        ("image + state", "none", None, True, True, 0),
        ("+ verb class", "verb_class", None, True, True, 1),
        ("+ subtask phrase", "sentence", "subtask_embedding", True, True, 2),
        ("+ instruction sentence", "sentence", "instruction_embedding", True, True, 3),
        ("instruction only", "sentence", "instruction_embedding", False, False, 3),
    )
    results = []
    goal_counts = np.asarray([np.sum(memory["task_index"] == t) for t in GOAL_TASKS], np.float64)
    chance = float(np.sum((goal_counts / goal_counts.sum()) ** 2))

    for feature_source, source_kind in feature_sources:
        train_v = _feature_variant(train, source_kind)
        memory_v = _feature_variant(memory, source_kind)
        query_v = _feature_variant(query, source_kind)
        # The wrist source is the 2048-D feature extracted from the wrist RGB
        # using the same frozen, language-free PaliGemma cache.
        if source_kind in ("original", "wrist"):
            selected_conditions = conditions[:1]
        else:
            selected_conditions = conditions
        for condition, subtask_mode, input_key, include_img, include_state, mode_seed in selected_conditions:
            if input_key:
                train_v["subtask_embedding"] = train_v[input_key]
                memory_v["subtask_embedding"] = memory_v[input_key]
                query_v["subtask_embedding"] = query_v[input_key]
            clip_seed = seed + HORIZON * 17 + 1000 * mode_seed
            torch.manual_seed(clip_seed)
            model, fit = fit_dual_encoder(
                train_v, train_b, use_subtask=subtask_mode != "none",
                subtask_mode=subtask_mode, nearby_positive=False,
                seed=clip_seed, epochs=CLIP_EPOCHS, patience=CLIP_PATIENCE,
                device=device, allowed_task_indices=SPLIT_B_TASKS,
                include_img=include_img, include_state=include_state)
            query_g = _query_embedding(model, fit, query_v, device=device,
                                       include_img=include_img, include_state=include_state)
            memory_g = _query_embedding(model, fit, memory_v, device=device,
                                        include_img=include_img, include_state=include_state)
            with torch.no_grad():
                memory_h = model.encode_skill(torch.as_tensor(
                    memory_b, dtype=torch.float32, device=device)).cpu().numpy()
            for retrieval, matrix in (("asymmetric g·h", memory_h),
                                      ("symmetric g·g", memory_g)):
                chosen = retrieve_embedding_top1(query_g, matrix)
                prediction = _predict_abstracted(
                    m1, fiii, memory_b[chosen], query["f"], query["s"],
                    t0=III_T0, seed=residual_seed, draws=4)
                metrics = retrieval_metrics(
                    query, memory, gold_b, memory_b, chosen, prediction, query_a,
                    early_mask=early_mask)
                results.append({"feature_source": feature_source,
                                "query_input": condition, "retrieval": retrieval,
                                "all": metrics["all"], "early": metrics["early"],
                                "clip_epochs": fit["epochs"],
                                "clip_validation_loss": fit["best_validation_loss"],
                                "modality_weights": model.modality_weights()})

    output_result = {
        "seed": seed, "split": "B; object+goal training, held-out goal episodes queried",
        "horizon": HORIZON, "heldout_goal_episodes": len(heldout),
        "train_rows": len(train_a), "goal_memory_rows": len(memory_a),
        "query_rows": len(query_a), "random_task_baseline": chance,
        "m1": m1.describe(), "fiii_best_epoch": fiii_fit["best_epoch"],
        "fiii_best_validation_loss": fiii_fit["best_validation_loss"],
        "camera_feature_cache": str(camera_cache),
        "camera_cache_contains_wrist": True,
        "initial_task_accuracy": initial_accuracy, "rows": results,
    }
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / (datetime.now(timezone.utc).strftime("task-ambiguity-k-only-%Y%m%dT%H%M%SZ")
                            + "-" + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    (output / "results.json").write_text(json.dumps(output_result, indent=2, ensure_ascii=False))
    (output / "report.md").write_text(_make_report(output_result))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--camera-cache", type=Path, default=DEFAULT_CAMERA_CACHE)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--pi05-base-params", type=Path, default=DEFAULT_PI05_BASE_PARAMS)
    parser.add_argument("--paligemma-tokenizer", type=Path, default=DEFAULT_PALIGEMMA_TOKENIZER)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    output = run(output_root=args.output_root, cache_dir=args.cache_dir,
                 dataset_dir=args.dataset, camera_cache=args.camera_cache,
                 annotations_path=args.annotations, params_path=args.pi05_base_params,
                 tokenizer_path=args.paligemma_tokenizer, seed=args.seed,
                 device_name=args.device)
    print(f"output: {output}")


if __name__ == "__main__":
    main()
