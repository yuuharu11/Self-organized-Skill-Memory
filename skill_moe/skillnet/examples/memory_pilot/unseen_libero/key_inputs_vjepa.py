#!/usr/bin/env python3
"""Exact-key V-JEPA inputs for KEY2 shared-latent retrieval.

The stored rows contain pooled base-camera and wrist-camera V-JEPA 2 ViT-L
tokens.  They are joined only by (task, episode, frame); no nearest-neighbour
fallback is permitted.  The optional cache builder reuses the existing
features128 cache and extracts only missing LIBERO-40 rows.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_ROOT = Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/vjepa")
DEFAULT_FEATURES = Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/latent-probe-01/features128/vjepa2-vitl-fpc64-256")
DEFAULT_KEY2 = Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-13")
DEFAULT_RUN = Path("/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01")
DEFAULT_DEV = Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/unseen-stride5")
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
TOKEN_COUNT = 64
TOKEN_DIM = 1024
CAMERA_GROUPS = {"base": 0, "wrist": 1}


def _key_tuple(value) -> tuple[int, int, int]:
    key = tuple(map(int, value))
    if len(key) != 3:
        raise ValueError(f"row key must be (task, episode, frame), got {key}")
    return key


def stable_row_map(keys: np.ndarray) -> dict[tuple[int, int, int], int]:
    result = {}
    for i, raw in enumerate(np.asarray(keys)):
        key = _key_tuple(raw)
        if key in result:
            raise ValueError(f"duplicate row key {key}")
        result[key] = i
    return result


def preceding_frame_keys(keys: np.ndarray, frame_step: int = 1) -> np.ndarray:
    """Return [previous,current] exact source-frame keys; never consult future rows."""
    if frame_step < 1:
        raise ValueError("frame_step must be positive")
    out = np.asarray(keys, dtype=np.int64).copy()
    if out.ndim != 2 or out.shape[1] != 3:
        raise ValueError("keys must have shape [N,3]")
    prev = out.copy()
    prev[:, 2] = np.maximum(0, prev[:, 2] - int(frame_step))
    return np.stack([prev, out], axis=1)


def vjepa_group_ids(batch: int, *, device=None) -> np.ndarray:
    """Base=0 and wrist=1 for concatenated [base patches, wrist patches]."""
    del device  # numpy API; callers can convert to a tensor on their device.
    if batch < 0:
        raise ValueError("batch must be non-negative")
    return np.tile(np.r_[np.zeros(TOKEN_COUNT, np.int64), np.ones(TOKEN_COUNT, np.int64)], (batch, 1))


def offset_vjepa_groups(groups: np.ndarray, *, vlm_group_count: int = 4) -> np.ndarray:
    """Move V-JEPA camera IDs above all VLM group IDs before token concatenation.

    Example: VLM IDs 0..3 => V-JEPA base=4, wrist=5. Token feature dimensions
    must be projected to a common width by the caller before concatenation.
    """
    if vlm_group_count < 1:
        raise ValueError("vlm_group_count must be positive")
    g = np.asarray(groups)
    if np.any((g < 0) | (g > 1)):
        raise ValueError("V-JEPA group IDs must be base=0 or wrist=1")
    return g.astype(np.int64) + int(vlm_group_count)


def _domain_dir(root: Path, domain: str) -> Path:
    if domain not in ("libero40", "libero90"):
        raise ValueError("domain must be 'libero40' or 'libero90'")
    return Path(root) / domain


def load_vjepa_tokens(keys: np.ndarray, *, cache_root: Path = DEFAULT_ROOT,
                      domain: str = "libero40") -> tuple[np.ndarray, np.ndarray]:
    """Load exact rows as [B,128,1024] tokens and [B,128] camera groups.

    The row order in ``keys`` is preserved. Duplicate requests are allowed;
    missing cache entries raise KeyError rather than being imputed.
    """
    keys = np.asarray(keys)
    if keys.ndim != 2 or keys.shape[1] != 3:
        raise ValueError("keys must have shape [B,3]")
    root = _domain_dir(Path(cache_root), domain)
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(root / "manifest.json")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("complete") is not True:
        raise ValueError(f"V-JEPA cache is not complete: {root}")
    row_keys = np.load(root / "keys.npy", mmap_mode="r", allow_pickle=False)
    mapping = stable_row_map(row_keys)
    wanted = [_key_tuple(k) for k in keys]
    missing = [k for k in wanted if k not in mapping]
    if missing:
        raise KeyError(f"{len(missing)} exact V-JEPA row keys missing; examples={missing[:5]}")
    indices = np.asarray([mapping[k] for k in wanted], np.int64)
    base = np.load(root / "base.npy", mmap_mode="r", allow_pickle=False)[indices]
    wrist = np.load(root / "wrist.npy", mmap_mode="r", allow_pickle=False)[indices]
    if base.shape != (len(keys), TOKEN_COUNT, TOKEN_DIM) or wrist.shape != base.shape:
        raise ValueError(f"bad cached feature shapes: base={base.shape} wrist={wrist.shape}")
    tokens = np.concatenate([base, wrist], axis=1).astype(np.float16, copy=False)
    groups = vjepa_group_ids(len(keys))
    if not np.isfinite(tokens).all():
        raise ValueError("non-finite V-JEPA tokens")
    return tokens, groups


def _sha_keys(keys: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(keys, dtype=np.int32).tobytes()).hexdigest()


def _cache_keys(path: Path) -> np.ndarray:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("model_id") != MODEL_ID or manifest.get("complete") is not True:
        raise ValueError(f"incompatible source feature cache: {path}")
    return np.load(path / "row_keys.npy", mmap_mode="r", allow_pickle=False)


def _source_features(key, source_caches: list[tuple[dict, Path]], *, domain: str):
    matches = []
    for map_, path in source_caches:
        if key in map_:
            idx = map_[key]
            base = np.load(path / "base.npy", mmap_mode="r", allow_pickle=False)[idx]
            wrist = np.load(path / "wrist.npy", mmap_mode="r", allow_pickle=False)[idx]
            matches.append((base, wrist))
    if not matches:
        return None
    first = matches[0]
    for other in matches[1:]:
        if not np.array_equal(first[0], other[0]) or not np.array_equal(first[1], other[1]):
            raise ValueError(f"overlapping source V-JEPA caches disagree for {domain}:{key}")
    return first


def _target_rows(key2_root: Path, run_root: Path, dev_rows_root: Path):
    """Build exact KEY2 unit-start, learned-eval, and selected Dev key sets."""
    key2 = Path(key2_root)
    bank_path = key2 / "smoke-all" / "bank.npz"
    if not bank_path.exists():
        raise FileNotFoundError(f"attempt-13 bank missing: {bank_path}")
    with np.load(bank_path, allow_pickle=False) as bank:
        unit_keys = np.asarray(bank["source"][:, :3], np.int32)
    if len(unit_keys) != 4516 or len(np.unique(unit_keys, axis=0)) != 4516:
        raise AssertionError("KEY2 memory-unit start keys are not the expected unique 4,516 rows")

    rows_root = Path(run_root) / "rows"
    row_data = {name: np.load(rows_root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                for name in ("task", "episode", "frame")}
    ref_path = Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/pi05-reference-01/selection.npz")
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)
    with np.load(ref_path, allow_pickle=False) as ref:
        learned_ids = np.asarray(ref["learned_row"], np.int64)
    if len(learned_ids) != 621:
        raise AssertionError(f"expected 621 learned-eval rows, got {len(learned_ids)}")
    learned_keys = np.column_stack([row_data["task"][learned_ids], row_data["episode"][learned_ids],
                                    row_data["frame"][learned_ids]]).astype(np.int32)

    dev_root = Path(dev_rows_root)
    dev_task = np.load(dev_root / "task.npy", mmap_mode="r", allow_pickle=False)
    dev_episode = np.load(dev_root / "episode.npy", mmap_mode="r", allow_pickle=False)
    dev_frame = np.load(dev_root / "frame.npy", mmap_mode="r", allow_pickle=False)
    dev_mask = np.asarray(np.load(dev_root / "is_dev.npy", mmap_mode="r", allow_pickle=False), bool)
    dev_keys = np.column_stack([dev_task[dev_mask], dev_episode[dev_mask], dev_frame[dev_mask]]).astype(np.int32)
    if len(dev_keys) != 217:
        raise AssertionError(f"expected 217 KEY2 Dev rows, got {len(dev_keys)}")
    return {"memory_unit_starts": unit_keys, "learned_eval": learned_keys, "dev_selection": dev_keys}


def _load_model(model_dir: Path, *, device: str, memory_fraction: float):
    import torch
    from transformers import AutoModel, AutoVideoProcessor
    torch.set_num_threads(8)
    if device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(float(memory_fraction), torch.device(device).index or 0)
    processor = AutoVideoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float16)
    model.eval().to(device)
    return model, processor


def _load_parquet_images(key, episode_map, parquet_cache):
    import pyarrow.parquet as pq
    from PIL import Image
    task, episode, frame = key
    try:
        rec = episode_map[(task, episode)]
    except KeyError as exc:
        raise KeyError(f"no source episode for exact row key {key}") from exc
    path = Path(rec["path"])
    if path not in parquet_cache:
        parquet_cache.clear()
        parquet_cache[path] = pq.read_table(path, columns=["image", "wrist_image"])
    table = parquet_cache[path]
    if frame >= len(table):
        raise IndexError(f"frame {frame} exceeds episode length {len(table)} for {key}")
    prev = max(0, int(frame) - 1)
    output = {}
    for field, name in (("image", "base"), ("wrist_image", "wrist")):
        frames = []
        for f in (prev, int(frame)):
            obj = table[field][f].as_py()
            with Image.open(io.BytesIO(obj["bytes"])) as im:
                rgb = im.convert("RGB")
                # This mirrors latent-probe V-JEPA manifest: source orientation,
                # then PIL bicubic resize to 128 before the HF video processor.
                rgb = rgb.resize((128, 128), resample=Image.Resampling.BICUBIC)
                frames.append(rgb.copy())
        output[name] = frames
    return output


def _encode_one(model, processor, images, *, device: str):
    import torch
    result = {}
    with torch.inference_mode():
        for camera in ("base", "wrist"):
            inputs = processor(videos=[images[camera]], return_tensors="pt")
            video = inputs["pixel_values_videos"].to(device)
            features = model(pixel_values_videos=video).last_hidden_state
            if features.ndim != 3 or features.shape[1:] != (256, TOKEN_DIM):
                raise ValueError(f"unexpected V-JEPA output {tuple(features.shape)}")
            grid = features.reshape(1, 16, 16, TOKEN_DIM)
            # Spatial 2x2 average, row-major, yielding 8x8 = 64 tokens.
            pooled = grid.reshape(1, 8, 2, 8, 2, TOKEN_DIM).mean((2, 4))
            result[camera] = pooled.reshape(1, TOKEN_COUNT, TOKEN_DIM)[0].float().cpu().numpy().astype(np.float16)
    return result["base"], result["wrist"]


def build_key2_cache(*, output_root: Path = DEFAULT_ROOT, feature_root: Path = DEFAULT_FEATURES,
                     key2_root: Path = DEFAULT_KEY2, run_root: Path = DEFAULT_RUN,
                     dev_rows_root: Path = DEFAULT_DEV, model_dir: Path | None = None,
                     device: str = "cuda:0", memory_fraction: float = .20,
                     verify_count: int = 8) -> dict:
    """Create target row caches, reusing exact keys and extracting only missing points."""
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not (output_root / "build_manifest.json").exists():
        # Allow resuming this builder's own interrupted, contract-tagged partial
        # output. Arbitrary files or a different contract remain protected.
        partial_manifests = list(output_root.glob("*/manifest.json"))
        if not partial_manifests or any(json.loads(p.read_text()).get("contract") != "key2-vjepa-input-v1"
                                        for p in partial_manifests):
            raise FileExistsError(f"refusing to overwrite unmanifested output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    requested = _target_rows(key2_root, run_root, dev_rows_root)
    cache_root = Path(feature_root)
    source_dirs = {
        "libero40": [cache_root / "libero40_fit"],
        "libero90": [cache_root / "libero90_dev"],
    }
    sources = {}
    for domain, dirs in source_dirs.items():
        sources[domain] = []
        for path in dirs:
            keys = _cache_keys(path)
            sources[domain].append((stable_row_map(keys), path))

    # Keep the domains separate because (task,episode,frame) can collide between suites.
    domain_requests = {
        "libero40": np.unique(np.concatenate([requested["memory_unit_starts"], requested["learned_eval"]]), axis=0).astype(np.int32),
        "libero90": requested["dev_selection"].astype(np.int32),
    }
    outputs = {}
    for domain, keys in domain_requests.items():
        outdir = output_root / domain
        outdir.mkdir(parents=True, exist_ok=True)
        manifest_path = outdir / "manifest.json"
        keys_path = outdir / "keys.npy"
        expected = [list(map(int, k)) for k in keys]
        if manifest_path.exists():
            prior = json.loads(manifest_path.read_text())
            if prior.get("keys") != expected:
                raise ValueError(f"existing {domain} cache manifest uses a different exact key list")
        else:
            temp = keys_path.with_suffix(".npy.tmp")
            with temp.open("wb") as f:
                np.save(f, keys, allow_pickle=False)
            temp.replace(keys_path)
            manifest_path.write_text(json.dumps({"contract": "key2-vjepa-input-v1", "domain": domain,
                "model_id": MODEL_ID, "keys": expected, "rows": len(keys), "complete": False}, indent=2)+"\n")
        base_path, wrist_path = outdir / "base.npy", outdir / "wrist.npy"
        shape = (len(keys), TOKEN_COUNT, TOKEN_DIM)
        for path in (base_path, wrist_path):
            if not path.exists():
                mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
                mm[:] = 0
                mm.flush()
                del mm
        base = np.lib.format.open_memmap(base_path, mode="r+", dtype=np.float16, shape=shape)
        wrist = np.lib.format.open_memmap(wrist_path, mode="r+", dtype=np.float16, shape=shape)
        done_path = outdir / "completed.npy"
        if done_path.exists():
            done = np.load(done_path, mmap_mode="r+", allow_pickle=False)
            if done.shape != (len(keys),):
                raise ValueError(f"bad resume bitmap for {domain}")
        else:
            done = np.lib.format.open_memmap(done_path, mode="w+", dtype=np.bool_, shape=(len(keys),))
            done[:] = False
        rowmap = stable_row_map(keys)
        source_present = [(_source_features(_key_tuple(raw), sources[domain], domain=domain) is not None)
                          for raw in keys]
        missing_indices = [i for i, present in enumerate(source_present) if not present and not done[i]]
        for i, raw in enumerate(keys):
            if done[i]:
                continue
            pair = _source_features(_key_tuple(raw), sources[domain], domain=domain)
            if pair is not None:
                base[i], wrist[i] = pair
                done[i] = True
        base.flush(); wrist.flush(); done.flush()

        extraction_result = {"requested": len(keys), "reused_existing": int(sum(source_present)),
                             "missing_in_existing_caches": int(len(keys)-sum(source_present)),
                             "extracted_or_resumed": int(sum((not present) and bool(done[i])
                                                              for i, present in enumerate(source_present))),
                             "missing_to_extract_this_run": len(missing_indices), "extracted_this_run": 0,
                             "source_cache_dirs": [str(p) for p in source_dirs[domain]]}
        if missing_indices:
            if domain != "libero40":
                raise KeyError(f"LIBERO-90 Dev cache unexpectedly misses {len(missing_indices)} rows")
            if model_dir is None:
                model_dir = cache_root.parent.parent / "data" / "vjepa2-vitl-fpc64-256"
            if not Path(model_dir).exists():
                raise FileNotFoundError(f"local V-JEPA model files not found: {model_dir}")
            model, processor = _load_model(Path(model_dir), device=device, memory_fraction=memory_fraction)
            data_manifest = json.loads((Path(run_root) / "data_manifest.json").read_text())
            episode_map = {(int(rec["task"]), int(rec["episode"])): rec for rec in data_manifest["episodes"]}
            parquet_cache = {}
            for j, i in enumerate(missing_indices):
                key = _key_tuple(keys[i])
                images = _load_parquet_images(key, episode_map, parquet_cache)
                base[i], wrist[i] = _encode_one(model, processor, images, device=device)
                done[i] = True
                if (j + 1) % 50 == 0:
                    base.flush(); wrist.flush(); done.flush()
            base.flush(); wrist.flush(); done.flush()
            extraction_result["extracted_this_run"] = len(missing_indices)
            del model, processor
        if not bool(np.asarray(done).all()):
            raise RuntimeError(f"incomplete V-JEPA feature cache for {domain}")
        del base, wrist, done
        manifest = {"contract": "key2-vjepa-input-v1", "domain": domain, "model_id": MODEL_ID,
            "rows": len(keys), "shape_per_camera": [TOKEN_COUNT, TOKEN_DIM], "dtype": "float16",
            "key_fields": ["task", "episode", "frame"], "key_sha256": _sha_keys(keys),
            "keys": expected, "camera_outputs": {"base": "base.npy", "wrist": "wrist.npy"},
            "input_temporal_order": ["immediately_previous_source_frame", "current_source_frame"],
            "frame_zero_rule": "duplicate current image as previous", "input_resize": "PIL BICUBIC 128x128",
            "image_orientation": "reuse feature-cache domain convention: LIBERO-40 no rotation; LIBERO-90 source manifest corrected HDF5 frame[::-1,::-1]",
            "spatial_pool": "2x2 mean from 16x16 patches to 8x8", "extraction": extraction_result,
            "complete": True}
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        outputs[domain] = manifest

    # Re-extract deterministic overlap rows from the source cache and compare; no silent drift.
    rng = np.random.default_rng(4101)
    parity = {}
    if verify_count:
        domain = "libero40"
        keys = domain_requests[domain]
        source_map, source_path = sources[domain][0]
        overlap = [tuple(map(int, k)) for k in keys if tuple(map(int, k)) in source_map]
        verify = [overlap[i] for i in rng.choice(len(overlap), min(verify_count, len(overlap)), replace=False)]
        model_dir = Path(model_dir or (cache_root.parent.parent / "data" / "vjepa2-vitl-fpc64-256"))
        model, processor = _load_model(model_dir, device=device, memory_fraction=memory_fraction)
        dm = json.loads((Path(run_root) / "data_manifest.json").read_text())
        episode_map = {(int(rec["task"]), int(rec["episode"])): rec for rec in dm["episodes"]}
        parquet_cache = {}
        target_dir = output_root / domain
        target_keys = np.load(target_dir / "keys.npy", mmap_mode="r", allow_pickle=False)
        target_map = stable_row_map(target_keys)
        tb = np.load(target_dir / "base.npy", mmap_mode="r", allow_pickle=False)
        tw = np.load(target_dir / "wrist.npy", mmap_mode="r", allow_pickle=False)
        errors = []
        for key in verify:
            pair = _encode_one(model, processor, _load_parquet_images(key, episode_map, parquet_cache), device=device)
            idx = target_map[key]
            # Compare the exact fp16 cache representation, reporting feature-space max absolute error.
            error_base = np.max(np.abs(pair[0].astype(np.float32) - tb[idx].astype(np.float32)))
            error_wrist = np.max(np.abs(pair[1].astype(np.float32) - tw[idx].astype(np.float32)))
            errors.append(float(max(error_base, error_wrist)))
        parity = {"rows_reextracted": len(errors), "max_abs_error": max(errors, default=None),
                  "exact_fp16_match": bool(errors and max(errors) == 0.), "keys": [list(k) for k in verify]}
        del model, processor, tb, tw
        if not errors or max(errors) > 5e-3:
            raise AssertionError(f"V-JEPA re-extraction parity failed: {parity}")
    build = {"contract": "key2-vjepa-build-v1", "model_id": MODEL_ID,
             "existing_feature_root": str(feature_root), "key2_attempt": str(key2_root),
             "row_sets": {name: int(len(keys)) for name, keys in requested.items()},
             "domain_caches": {name: {"rows": int(value["rows"]),
                 "shape_per_camera": value["shape_per_camera"], "dtype": value["dtype"],
                 "key_sha256": value["key_sha256"], "extraction": value["extraction"]}
                 for name, value in outputs.items()},
             "reextraction_parity": parity,
             "group_ids": {"base": 0, "wrist": 1}, "combined_vlm_rule": "offset V-JEPA group IDs by number of VLM groups (usually +4)",
             "complete": True}
    encoded = json.dumps(build, indent=2)+"\n"
    (output_root / "build_manifest.json").write_text(encoded)
    (output_root / "manifest.json").write_text(encoded)
    return build


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--key2-root", type=Path, default=DEFAULT_KEY2)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dev-rows-root", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=.20)
    parser.add_argument("--verify-count", type=int, default=8)
    args = parser.parse_args(argv)
    result = build_key2_cache(output_root=args.output_root, feature_root=args.feature_root,
        key2_root=args.key2_root, run_root=args.run_root, dev_rows_root=args.dev_rows_root,
        model_dir=args.model_dir, device=args.device, memory_fraction=args.memory_fraction,
        verify_count=args.verify_count)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
