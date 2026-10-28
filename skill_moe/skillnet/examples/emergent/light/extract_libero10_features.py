"""Cache frozen pi0.5 visual features for task IDs 0-9 in LIBERO-40."""
from __future__ import annotations

import argparse
import io
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .extract import READ_COLUMNS, _image_bytes
from .libero10_features import _action_chunks, _motion, initial_state_id

DEFAULT_DATA = Path("/work/.skillnet-state/data/lerobot/emergent/libero_40_unsegmented")
DEFAULT_CHECKPOINT = Path("/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_base/params")
DEFAULT_OUTPUT = Path("/work/.skillnet-state/outputs/memory-pilot")
STRIDE = 5
SPLIT_SEED = 20261026


def _read_episode(path: Path, meta: dict[str, Any]) -> dict[str, Any] | None:
    table = pq.read_table(path, columns=READ_COLUMNS)
    task_ids = np.asarray(table["task_index"].to_numpy())
    task_id = int(task_ids[0])
    if task_id not in range(10):
        return None
    steps = np.asarray(table["frame_index"].to_numpy(), dtype=np.int32)
    state = np.asarray(table["state"].combine_chunks().values.to_numpy(), dtype=np.float32).reshape(len(steps), -1)
    action = np.asarray(table["actions"].combine_chunks().values.to_numpy(), dtype=np.float32).reshape(len(steps), -1)
    return {
        "episode_id": f"libero40-{int(meta['episode_index']):06d}", "episode_index": int(meta["episode_index"]),
        "task_index": task_id, "suite": "libero10", "task": str(meta["tasks"][0]), "steps": steps,
        "state": state, "action": action,
        "images": [_image_bytes(table["image"], i) for i in range(len(steps))],
        "wrist_images": [_image_bytes(table["wrist_image"], i) for i in range(len(steps))],
    }


def _initial_states(dataset: Path) -> list[dict[str, Any]]:
    metas = {int(d["episode_index"]): d for d in
             (json.loads(line) for line in (dataset / "meta/episodes.jsonl").open(encoding="utf-8"))}
    output = []
    for p in sorted((dataset / "data").glob("chunk-*/episode_*.parquet")):
        ix = int(p.stem.split("_")[-1])
        meta = metas[ix]
        t = pq.read_table(p, columns=["task_index", "state"])
        task_id = int(t["task_index"][0].as_py())
        if task_id not in range(10):
            continue
        first_state = np.asarray(t["state"][0].as_py(), dtype="<f4")
        output.append({"episode_index": ix, "task_index": task_id,
                       "init_state_id": initial_state_id(first_state, task_id), "path": p, "meta": meta})
    return output


def split_initial_states(inventory: list[dict[str, Any]], *, seed: int = SPLIT_SEED) -> dict[str, str]:
    """Make a deterministic, task-stratified 75/25 split by initial-state group."""
    rng = np.random.default_rng(seed)
    by_task: dict[int, dict[str, list[int]]] = {}
    for entry in inventory:
        by_task.setdefault(entry["task_index"], {}).setdefault(entry["init_state_id"], []).append(entry["episode_index"])
    total_groups = sum(len(groups) for groups in by_task.values())
    target_query = int(round(0.25 * total_groups))
    query_counts = {task_id: int(np.floor(0.25 * len(groups))) for task_id, groups in by_task.items()}
    remainder = target_query - sum(query_counts.values())
    ties = {task_id: float(rng.random()) for task_id in by_task}
    ranked = sorted(by_task, key=lambda task_id: (-(0.25 * len(by_task[task_id]) - query_counts[task_id]), ties[task_id]))
    for task_id in ranked[:remainder]:
        query_counts[task_id] += 1
    assignment: dict[str, str] = {}
    for task_id, groups in sorted(by_task.items()):
        ids = sorted(groups)
        if len(ids) < 2:
            raise ValueError(f"task {task_id} has fewer than two distinct initial states; cannot make store/query split")
        order = rng.permutation(len(ids))
        query_count = min(len(ids) - 1, max(1, query_counts[task_id]))
        query_ids = {ids[int(i)] for i in order[:query_count]}
        for init_id in ids:
            name = "query" if init_id in query_ids else "store"
            for episode_index in groups[init_id]:
                assignment[f"libero40-{episode_index:06d}"] = name
    return assignment


def repartition_cached_features(source_root: Path, dataset: Path, checkpoint: Path, output_root: Path,
                                *, seed: int = SPLIT_SEED) -> Path:
    """Repack already extracted NPZs under the exact 75/25 split, without VLM inference."""
    inventory = _initial_states(dataset)
    split = split_initial_states(inventory, seed=seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    root = output_root / f"libero10-features-{run_id}"
    root.mkdir(parents=True, exist_ok=False)
    source_feature = source_root / "feature-cache" / "episodes"
    source_camera = source_root / "camera-features" / "episodes"
    source_language = source_root / "camera-language-features" / "episodes"
    target_feature = root / "feature-cache" / "episodes"
    target_camera = root / "camera-features" / "episodes"
    target_language = root / "camera-language-features" / "episodes"
    for directory in (target_feature, target_camera, target_language):
        directory.mkdir(parents=True, exist_ok=False)
    for entry in inventory:
        episode_id = f"libero40-{entry['episode_index']:06d}"
        with np.load(source_feature / f"{episode_id}.npz", allow_pickle=False) as data:
            payload = {name: data[name] for name in data.files}
        payload["split"] = np.asarray(split[episode_id])
        with (target_feature / f"{episode_id}.npz").open("xb") as f:
            np.savez_compressed(f, **payload)
        os.link(source_camera / f"{episode_id}.npz", target_camera / f"{episode_id}.npz")
        os.link(source_language / f"{episode_id}.npz", target_language / f"{episode_id}.npz")
    return finalize_cache(dataset, checkpoint, root, seed=seed, empty_prompt_check_passed=True)


def finalize_cache(dataset: Path, checkpoint: Path, root: Path, *, seed: int = SPLIT_SEED,
                   empty_prompt_check_passed: bool = True) -> Path:
    """Write manifests for a complete cache tree without running the VLM again."""
    inventory = _initial_states(dataset)
    split = split_initial_states(inventory, seed=seed)
    feature_root, camera_root, camera_lang_root = root / "feature-cache", root / "camera-features", root / "camera-language-features"
    feature_manifest, camera_manifest, language_manifest, episodes = [], [], [], []
    for entry in inventory:
        episode_id = f"libero40-{entry['episode_index']:06d}"
        split_name = split[episode_id]
        init_id = entry["init_state_id"]
        feature_cache = feature_root / "episodes" / f"{episode_id}.npz"
        camera_cache = camera_root / "episodes" / f"{episode_id}.npz"
        lang_cache = camera_lang_root / "episodes" / f"{episode_id}.npz"
        if not (feature_cache.exists() and camera_cache.exists() and lang_cache.exists()):
            raise FileNotFoundError(f"incomplete LIBERO-10 episode cache: {episode_id}")
        with np.load(feature_cache, allow_pickle=False) as d:
            n_rows, n_frames = len(d["step"]), len(d["env_step"])
            if d["f"].shape != (n_rows, 2048) or d["k"].shape != (n_rows, 2048):
                raise ValueError(f"{episode_id}: f/k shape differs from expected PaliGemma width 2048")
            if d["f"].dtype != np.float16 or d["k"].dtype != np.float16:
                raise ValueError(f"{episode_id}: f/k dtype must be float16")
            if d["A_H25"].shape != (n_rows, 25, 7) or d["A_H10"].shape != (n_rows, 10, 7):
                raise ValueError(f"{episode_id}: action chunk shape mismatch")
            if d["motion_H25"].shape != (n_rows, 25, 9) or d["motion_H10"].shape != (n_rows, 10, 9):
                raise ValueError(f"{episode_id}: evaluation motion shape mismatch")
        with np.load(camera_cache, allow_pickle=False) as d:
            for key in ("base", "wrist"):
                if d[key].shape != (n_rows, 2048) or d[key].dtype != np.float16:
                    raise ValueError(f"{episode_id}: camera {key} shape/dtype mismatch")
        with np.load(lang_cache, allow_pickle=False) as d:
            for key in ("base_lang", "wrist_lang"):
                if d[key].shape != (n_rows, 2048) or d[key].dtype != np.float16:
                    raise ValueError(f"{episode_id}: language camera {key} shape/dtype mismatch")
        row = {"episode_id": episode_id, "suite": "libero10", "task": entry["meta"]["tasks"][0],
               "task_index": entry["task_index"], "init_state_id": init_id, "split": split_name,
               "n_rows": n_rows, "n_frames": n_frames, "cache": str(feature_cache),
               "camera_cache": str(camera_cache), "camera_language_cache": str(lang_cache)}
        episodes.append(row)
        feature_manifest.append({k: row[k] for k in ("episode_id", "suite", "task", "task_index", "n_rows", "n_frames", "cache", "init_state_id", "split")})
        camera_manifest.append({k: row[k] for k in ("episode_id", "suite", "task", "task_index", "n_rows", "camera_cache", "init_state_id", "split")})
        language_manifest.append({k: row[k] for k in ("episode_id", "suite", "task", "task_index", "n_rows", "camera_language_cache", "init_state_id", "split")})
    common = {"dataset": str(dataset), "checkpoint": str(checkpoint), "stride": STRIDE,
              "seed": int(seed), "split_unit": "task-scoped hash of rounded initial proprio state (xyz, axis-angle, gripper qpos)",
              "store_query_ratio": "75/25 by initial-state groups, stratified by task", "episodes": feature_manifest}
    split_report = {"seed": int(seed), "by_task": {}, "query_init_state_ids": [], "store_init_state_ids": [],
                    "no_language_prompt_invariance_checked": bool(empty_prompt_check_passed)}
    for task_id in range(10):
        entries = [e for e in inventory if e["task_index"] == task_id]
        ids = {e["init_state_id"]: split[f"libero40-{e['episode_index']:06d}"] for e in entries}
        qids = sorted(i for i, value in ids.items() if value == "query")
        sids = sorted(i for i, value in ids.items() if value == "store")
        split_report["query_init_state_ids"].extend(qids)
        split_report["store_init_state_ids"].extend(sids)
        split_report["by_task"][str(task_id)] = {
            "episodes": len(entries), "init_states": len(ids), "store_init_states": len(sids), "query_init_states": len(qids),
            "store_episodes": sum(split[f"libero40-{e['episode_index']:06d}"] == "store" for e in entries),
            "query_episodes": sum(split[f"libero40-{e['episode_index']:06d}"] == "query" for e in entries),
        }
    for path, data in (
        (feature_root / "manifest.json", {**common, "cache_format": "light.extract episode NPZ; f/k float16; action/proprio float32; H25/H10 chunks and G2-compatible evaluation motion included"}),
        (camera_root / "manifest.json", {**common, "episodes": camera_manifest, "features": "base/wrist no-language, same image-only PaliGemma pooling as compare_keys.py"}),
        (camera_lang_root / "manifest.json", {**common, "episodes": language_manifest, "features": "base/wrist language-conditioned image-token pooling as compare_language_keys.py"}),
        (root / "split.json", split_report),
        (root / "manifest.json", {**common, "feature_cache": str(feature_root), "camera_cache": str(camera_root),
            "camera_language_cache": str(camera_lang_root),
            "no_language_definition": "image-only: task text tokens are omitted from the PaliGemma prefix and attention mask",
            "empty_prompt_check": "same image batch evaluated with two different token sequences while has_language=False; outputs matched",
            "split": split_report, "episodes": episodes}),
    ):
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous != data:
                raise FileExistsError(f"refusing to overwrite nonmatching cache metadata: {path}")
            continue
        with path.open("x", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    return root


def _feature_extractor(checkpoint: Path):
    import jax
    import jax.numpy as jnp
    from openpi.models import pi0_config
    from openpi.models.model import restore_params
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import image_tools

    cfg = pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m",
                               action_horizon=25, max_token_len=200, dtype="bfloat16")
    model = cfg.load(restore_params(checkpoint, dtype=jnp.bfloat16))
    resize = jax.jit(lambda x: image_tools.resize_with_pad(x, 224, 224))
    tokenizer = PaligemmaTokenizer(200)

    def infer(image_views, tokens, token_mask, has_language):
        image_tokens = [model.PaliGemma.img(image, train=False)[0] for image in image_views]
        prefix = jnp.concatenate(image_tokens, axis=1)
        image_prefix_len = prefix.shape[1]
        masks = [jnp.ones((prefix.shape[0], image_prefix_len), dtype=jnp.bool_)]
        ar = [jnp.zeros((image_prefix_len,), dtype=jnp.bool_)]
        if has_language:
            text = model.PaliGemma.llm(tokens, method="embed")
            prefix = jnp.concatenate([prefix, text], axis=1)
            masks.append(token_mask)
            ar.append(jnp.zeros((text.shape[1],), dtype=jnp.bool_))
        input_mask = jnp.concatenate(masks, axis=1)
        ar_mask = jnp.concatenate(ar)
        attention = jnp.cumsum(ar_mask)[None, :, None] <= jnp.cumsum(ar_mask)[None, None, :]
        attention = attention & input_mask[:, None, :] & input_mask[:, :, None]
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (out, _), _ = model.PaliGemma.llm([prefix, None], mask=attention, positions=positions,
                                          adarms_cond=[None, None])
        # Both the combined and one-camera features pool image tokens only.
        return jnp.mean(out[:, :image_prefix_len, :].astype(jnp.float32), axis=1)

    infer_jit = jax.jit(infer, static_argnums=(3,))

    def preprocess(image_bytes):
        raw = [np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8) for b in image_bytes]
        x = np.asarray(resize(jnp.asarray(np.stack(raw))))
        return jnp.asarray(x, dtype=jnp.float32) / 127.5 - 1.0

    def extract(base_bytes, wrist_bytes, tasks):
        base_image, wrist_image = preprocess(base_bytes), preprocess(wrist_bytes)
        toks, masks = zip(*(tokenizer.tokenize(task, None) for task in tasks))
        tokens, token_mask = jnp.asarray(np.stack(toks)), jnp.asarray(np.stack(masks))
        # This follows light.extract._pooler exactly: base and wrist image token
        # sequences are concatenated, and f/k are the first 512 image tokens.
        f = infer_jit((base_image, wrist_image), tokens, token_mask, True)
        k = infer_jit((base_image, wrist_image), tokens, token_mask, False)
        # Match compare_keys' image-only camera pooling and compare_language_keys'
        # camera image-token pooling with the instruction appended after images.
        camera_images = jnp.concatenate([base_image, wrist_image], axis=0)
        camera_tasks = list(tasks) + list(tasks)
        camera_toks, camera_masks = zip(*(tokenizer.tokenize(task, None) for task in camera_tasks))
        ct, cm = jnp.asarray(np.stack(camera_toks)), jnp.asarray(np.stack(camera_masks))
        no_language = infer_jit((camera_images,), ct, cm, False)
        with_language = infer_jit((camera_images,), ct, cm, True)
        b = len(tasks)
        no_language = no_language.reshape(2, b, -1).transpose(1, 0, 2)
        with_language = with_language.reshape(2, b, -1).transpose(1, 0, 2)
        return tuple(np.asarray(x, dtype=np.float32) for x in (f, k)), no_language, with_language

    def empty_prompt_check(base_bytes, wrist_bytes, tasks):
        base_image, wrist_image = preprocess(base_bytes), preprocess(wrist_bytes)
        real, masks = zip(*(tokenizer.tokenize(task, None) for task in tasks))
        other, _ = tokenizer.tokenize("unrelated instruction must not affect image-only features", None)
        tokens = np.stack(real)
        changed = np.repeat(other[None, :], len(tasks), axis=0)
        token_mask = np.stack(masks)
        # Same image and batch shapes, but different token IDs. The false static
        # branch must omit text from both the prefix and its attention mask.
        a = infer_jit((base_image, wrist_image), jnp.asarray(tokens), jnp.asarray(token_mask), False)
        b = infer_jit((base_image, wrist_image), jnp.asarray(changed), jnp.asarray(token_mask), False)
        return np.asarray(a), np.asarray(b)

    return extract, empty_prompt_check


def extract_libero10(dataset: Path, checkpoint: Path, output_root: Path, *, batch_size: int = 2,
                     seed: int = SPLIT_SEED) -> Path:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    inventory = _initial_states(dataset)
    split = split_initial_states(inventory, seed=seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    root = output_root / f"libero10-features-{run_id}"
    root.mkdir(parents=True, exist_ok=False)
    feature_root, camera_root, camera_lang_root = root / "feature-cache", root / "camera-features", root / "camera-language-features"
    for p in (feature_root / "episodes", camera_root / "episodes", camera_lang_root / "episodes"):
        p.mkdir(parents=True, exist_ok=False)
    extract, empty_prompt_check = _feature_extractor(checkpoint)
    metas = {entry["episode_index"]: entry for entry in inventory}
    feature_manifest, camera_manifest, language_manifest, episodes = [], [], [], []
    checked_empty_prompt = False
    started = time.monotonic()

    # Process episodes sequentially and keep only the current decoded images in memory.
    for n, entry in enumerate(inventory):
        ep = _read_episode(entry["path"], entry["meta"])
        if ep is None:
            raise AssertionError("task filter changed between inventory and extraction")
        sampled = np.arange(0, len(ep["steps"]), STRIDE, dtype=np.int32)
        f_rows, k_rows, no_lang_rows, lang_rows = [], [], [], []
        for lo in range(0, len(sampled), batch_size):
            ids = sampled[lo:lo + batch_size]
            real_n = len(ids)
            if real_n < batch_size:
                ids = np.pad(ids, (0, batch_size - real_n), mode="edge")
            base_bytes = [ep["images"][i] for i in ids]
            wrist_bytes = [ep["wrist_images"][i] for i in ids]
            task_batch = [ep["task"]] * len(ids)
            (f, k), nolang, lang = extract(base_bytes, wrist_bytes, task_batch)
            f_rows.extend(f[:real_n]); k_rows.extend(k[:real_n])
            no_lang_rows.extend(nolang[:real_n]); lang_rows.extend(lang[:real_n])
            if not checked_empty_prompt:
                direct_a, direct_b = empty_prompt_check(base_bytes, wrist_bytes, task_batch)
                if not np.allclose(direct_a, direct_b, rtol=0, atol=1e-6):
                    raise AssertionError("language-free k changed when prompt token IDs changed")
                if not np.allclose(direct_a, k, rtol=1e-5, atol=1e-5):
                    raise AssertionError("cached k differs from the independently checked image-only output")
                checked_empty_prompt = True

        h25, m25 = _action_chunks(ep["action"], ep["steps"][sampled], 25)
        h10, m10 = _action_chunks(ep["action"], ep["steps"][sampled], 10)
        motion25, mm25 = _motion(ep["state"], ep["steps"][sampled], 25)
        motion10, mm10 = _motion(ep["state"], ep["steps"][sampled], 10)
        split_name = split[ep["episode_id"]]
        init_id = entry["init_state_id"]
        base_cache = feature_root / "episodes" / f"{ep['episode_id']}.npz"
        camera_cache = camera_root / "episodes" / f"{ep['episode_id']}.npz"
        lang_cache = camera_lang_root / "episodes" / f"{ep['episode_id']}.npz"
        payload = {
            "step": ep["steps"][sampled], "env_step": ep["steps"], "proprio": ep["state"][sampled],
            "action": ep["action"], "env_state": ep["state"],
            "f": np.asarray(f_rows, np.float16), "k": np.asarray(k_rows, np.float16),
            "A_H25": h25, "A_mask_H25": m25, "A_H10": h10, "A_mask_H10": m10,
            "motion_H25": motion25, "motion_mask_H25": mm25,
            "motion_H10": motion10, "motion_mask_H10": mm10,
            "task_index": np.asarray(ep["task_index"], np.int32),
            "init_state_id": np.asarray(init_id), "split": np.asarray(split_name),
        }
        with base_cache.open("xb") as fobj:
            np.savez_compressed(fobj, **payload)
        cam = np.asarray(no_lang_rows, np.float16)
        with camera_cache.open("xb") as fobj:
            np.savez_compressed(fobj, step=ep["steps"][sampled], base=cam[:, 0], wrist=cam[:, 1])
        cam_lang = np.asarray(lang_rows, np.float16)
        with lang_cache.open("xb") as fobj:
            np.savez_compressed(fobj, step=ep["steps"][sampled], base_lang=cam_lang[:, 0], wrist_lang=cam_lang[:, 1])
        row = {"episode_id": ep["episode_id"], "suite": ep["suite"], "task": ep["task"],
               "task_index": ep["task_index"], "init_state_id": init_id, "split": split_name,
               "n_rows": len(sampled), "n_frames": len(ep["steps"]),
               "cache": str(base_cache), "camera_cache": str(camera_cache), "camera_language_cache": str(lang_cache)}
        episodes.append(row)
        feature_manifest.append({k: row[k] for k in ("episode_id", "suite", "task", "task_index", "n_rows", "n_frames", "cache", "init_state_id", "split")})
        camera_manifest.append({k: row[k] for k in ("episode_id", "suite", "task", "task_index", "n_rows", "camera_cache", "init_state_id", "split")})
        language_manifest.append({k: row[k] for k in ("episode_id", "suite", "task", "task_index", "n_rows", "camera_language_cache", "init_state_id", "split")})
        if (n + 1) % 10 == 0 or n == 0:
            print(f"LIBERO-10 feature extraction: episodes={n + 1}/{len(inventory)} rows={sum(x['n_rows'] for x in episodes)} elapsed_min={(time.monotonic() - started) / 60:.1f}", flush=True)
        del ep

    finalize_cache(dataset, checkpoint, root, seed=seed, empty_prompt_check_passed=checked_empty_prompt)
    print(f"LIBERO-10 feature cache: {root}", flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--finalize-root", type=Path, help="write missing manifests for an already complete episode cache")
    parser.add_argument("--repartition-from", type=Path, help="repack an extracted cache to the exact task-stratified 75/25 split without GPU inference")
    args = parser.parse_args()
    if args.finalize_root is not None:
        finalize_cache(args.dataset, args.checkpoint, args.finalize_root, seed=args.seed)
        print(f"LIBERO-10 feature cache finalized: {args.finalize_root}", flush=True)
        return
    if args.repartition_from is not None:
        root = repartition_cached_features(args.repartition_from, args.dataset, args.checkpoint, args.output_root, seed=args.seed)
        print(f"LIBERO-10 feature cache repartitioned: {root}", flush=True)
        return
    extract_libero10(args.dataset, args.checkpoint, args.output_root, batch_size=args.batch_size, seed=args.seed)


if __name__ == "__main__":
    main()
