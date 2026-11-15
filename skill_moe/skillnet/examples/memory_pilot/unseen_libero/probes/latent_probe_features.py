#!/usr/bin/env python3
"""Frozen V-JEPA feature extraction for latent-probe-data-v1.

The extractor consumes only the oriented image and identity arrays prepared by
``latent_probe_data.py``. It writes keyed fp16 memmaps; action/state labels are
never read. Each camera is processed independently as a two-frame clip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


DEFAULT_DATA = Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/latent-probe-01/data")
DEFAULT_OUTPUT = Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/latent-probe-01/features/vjepa2-vitl-fpc64-256")
DEFAULT_OUTPUT_128 = Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/latent-probe-01/features128/vjepa2-vitl-fpc64-256")
SPLITS = ("libero40_fit", "libero90_dev", "libero90_eval")
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


def _sha256(path: Path, chunk=1 << 20):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def pool_2x2_tokens(tokens: np.ndarray) -> np.ndarray:
    """Pool a 16x16 patch grid to 8x8 tokens, preserving embedding width."""
    x = np.asarray(tokens)
    if x.ndim != 3 or x.shape[1] != 256:
        raise ValueError(f"expected [B,256,D] tokens, got {x.shape}")
    b, _, d = x.shape
    return x.reshape(b, 8, 2, 8, 2, d).mean(axis=(2, 4)).reshape(b, 64, d)


def freeze_encoder(model):
    """Put a feature encoder in inference mode and disable all parameter grads."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError("encoder contains trainable parameters")
    return model


def resize_pil_bicubic(image: np.ndarray, size=128) -> np.ndarray:
    """Match worker.py's PIL RGB resize with its implicit BICUBIC default."""
    from PIL import Image

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB HWC image, got {image.shape} {image.dtype}")
    return np.asarray(Image.fromarray(np.ascontiguousarray(image)).resize(
        (int(size), int(size)), resample=Image.Resampling.BICUBIC))


def _resize_clip(clip, size):
    if size is None:
        return clip
    return np.stack([resize_pil_bicubic(frame, size) for frame in clip])


def preprocessor_tensor_stats(pixel_values):
    """Mean, variance, and mean squared adjacent-pixel energy after HF preprocessing."""
    x = np.asarray(pixel_values, np.float32)
    if x.ndim < 4:
        raise ValueError(f"expected batched video/image tensor, got {x.shape}")
    dx = np.diff(x, axis=-1)
    dy = np.diff(x, axis=-2)
    return {"shape": list(x.shape), "mean": float(x.mean()), "variance": float(x.var()),
            "high_frequency_energy_x": float(np.mean(dx * dx)),
            "high_frequency_energy_y": float(np.mean(dy * dy)),
            "high_frequency_energy_mean": float((np.mean(dx * dx) + np.mean(dy * dy)) / 2)}


def audit_resolution_preprocessing(data_root=DEFAULT_DATA, model_dir=None, *, sample_rows=64):
    """Compare 256→processor, 256→128→processor, and native-128→processor.

    Samples are deterministic and split-level rather than paired across tasks;
    LIBERO-40 and LIBERO-90 do not share row identities.
    """
    from transformers import AutoVideoProcessor

    model_path = Path(model_dir or (Path(data_root) / "vjepa2-vitl-fpc64-256"))
    processor = AutoVideoProcessor.from_pretrained(model_path, local_files_only=True)
    fit_root, _, fit_keys, fit_imgs = _open_split(data_root, "libero40_fit")
    eval_root, _, eval_keys, eval_imgs = _open_split(data_root, "libero90_eval")
    n = min(int(sample_rows), len(fit_keys), len(eval_keys))
    fit_ix = np.linspace(0, len(fit_keys) - 1, n, dtype=np.int64)
    eval_ix = np.linspace(0, len(eval_keys) - 1, n, dtype=np.int64)
    stats = {}
    for cam in ("base", "wrist"):
        now, prev = f"image_{cam}", f"image_{cam}_prev"
        variants = {
            "libero40_native_256": [_resize_clip(np.stack([fit_imgs[prev][i], fit_imgs[now][i]]), None) for i in fit_ix],
            "libero40_resized_128": [_resize_clip(np.stack([fit_imgs[prev][i], fit_imgs[now][i]]), 128) for i in fit_ix],
            "libero90_native_128": [np.stack([eval_imgs[prev][i], eval_imgs[now][i]]) for i in eval_ix],
        }
        stats[cam] = {}
        for variant, clips in variants.items():
            tensor = processor(videos=clips, return_tensors="pt")["pixel_values_videos"]
            stats[cam][variant] = preprocessor_tensor_stats(tensor.numpy())
    # Exact equivalence of worker.py's implicit PIL default and our explicit filter.
    from PIL import Image
    check = np.asarray(fit_imgs["image_base"][fit_ix[0]])
    implicit = np.asarray(Image.fromarray(np.ascontiguousarray(check)).resize((128, 128)))
    explicit = resize_pil_bicubic(check, 128)
    if not np.array_equal(implicit, explicit):
        raise AssertionError("explicit PIL BICUBIC differs from worker.py's resize default")
    return {"sample_rows_per_split": n, "sample_keys_libero40": fit_keys[fit_ix].tolist(),
            "sample_keys_libero90_eval": eval_keys[eval_ix].tolist(), "processor": str(model_path),
            "image_filter": "PIL.Image.Resampling.BICUBIC (pixel-exact to worker.py default)",
            "same_source_image_resize_pixel_check": True, "statistics": stats}


def _open_split(data_root: Path, split: str):
    root = Path(data_root) / split
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise RuntimeError(f"input split is not complete: {root}")
    keys = np.stack([np.load(root / f"{k}.npy", mmap_mode="r", allow_pickle=False)
                     for k in ("task", "episode", "frame")], axis=1)
    n = int(manifest["rows"])
    if keys.shape != (n, 3):
        raise ValueError(f"bad key shape in {root}: {keys.shape}")
    imgs = {name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("image_base", "image_base_prev", "image_wrist", "image_wrist_prev")}
    if any(len(v) != n for v in imgs.values()):
        raise ValueError(f"image/key row mismatch in {root}")
    return root, manifest, keys, imgs


def _digest_keys(keys: np.ndarray):
    return hashlib.sha256(np.asarray(keys, dtype=np.int32).tobytes()).hexdigest()


def extract_vjepa(data_root=DEFAULT_DATA, output_root=DEFAULT_OUTPUT, *, model_dir=None,
                  batch_size=8, pool_tokens=False, memory_fraction=0.25,
                  max_rows: int | None = None, input_resolution: int | None = None,
                  row_indices_by_split=None):
    """Extract all splits, resuming from per-row completion maps.

    ``max_rows`` is provided for smoke tests. It is intentionally not exposed
    as a defaulted truncation in the production CLI.
    """
    import torch
    from transformers import AutoModel, AutoVideoProcessor

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    if not torch.cuda.is_available():
        raise RuntimeError("V-JEPA extraction requires CUDA")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    torch.cuda.set_per_process_memory_fraction(float(memory_fraction))
    model_path = Path(model_dir or (Path(data_root) / "vjepa2-vitl-fpc64-256"))
    if not model_path.exists():
        raise FileNotFoundError(f"local V-JEPA weights are required: {model_path}")
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16,
                                      local_files_only=True).cuda().eval()
    model = freeze_encoder(model)
    processor = AutoVideoProcessor.from_pretrained(model_path, local_files_only=True)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "manifest.json").exists():
        raise FileExistsError(f"complete feature output already exists: {output_root}")
    started = time.time()
    summaries = {}

    for split in SPLITS:
        src, source_manifest, keys, imgs = _open_split(data_root, split)
        n_total = len(keys)
        if row_indices_by_split is not None and split in row_indices_by_split:
            selected_rows = np.asarray(row_indices_by_split[split], dtype=np.int64)
            if selected_rows.ndim != 1 or np.any(selected_rows < 0) or np.any(selected_rows >= n_total):
                raise ValueError(f"invalid selected source row indices for {split}")
            if len(np.unique(selected_rows)) != len(selected_rows):
                raise ValueError(f"duplicate selected source row indices for {split}")
        else:
            selected_rows = np.arange(min(n_total, int(max_rows)) if max_rows is not None else n_total,
                                      dtype=np.int64)
        n = len(selected_rows)
        selected_keys = keys[selected_rows]
        dst = output_root / split
        dst.mkdir(parents=True, exist_ok=True)
        key_hash = _digest_keys(selected_keys)
        tokens = 64 if pool_tokens else 256
        feature_spec = {cam: (dst / f"{cam}.npy") for cam in ("base", "wrist")}
        progress_path = dst / "completed.npy"
        meta_path = dst / "extracting.json"
        shape = (n, tokens, 1024)
        # Restart only a compatible partial extraction. Never truncate or
        # repurpose files created for a different source/configuration.
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            expected = {"rows": n, "key_sha256": key_hash, "shape": list(shape),
                        "pool_tokens": bool(pool_tokens), "model_id": MODEL_ID,
                        "source_row_indices_sha256": hashlib.sha256(selected_rows.tobytes()).hexdigest(),
                        "input_resolution": input_resolution}
            if any(meta.get(k) != v for k, v in expected.items()):
                raise ValueError(f"partial extraction metadata mismatch at {dst}")
        else:
            if any(p.exists() for p in [*feature_spec.values(), progress_path]):
                raise FileExistsError(f"orphan extraction arrays exist without metadata: {dst}")
            meta_path.write_text(json.dumps({"rows": n, "key_sha256": key_hash,
                "shape": list(shape), "pool_tokens": bool(pool_tokens), "model_id": MODEL_ID,
                "source_row_indices_sha256": hashlib.sha256(selected_rows.tobytes()).hexdigest(),
                "input_resolution": input_resolution,
                "source_split_manifest": str(src / "manifest.json"),
                "source_split_manifest_sha256": _sha256(src / "manifest.json")}, indent=2) + "\n")
        out_maps = {}
        for cam, path in feature_spec.items():
            if not path.exists():
                out_maps[cam] = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
            else:
                out_maps[cam] = np.load(path, mmap_mode="r+", allow_pickle=False)
                if out_maps[cam].shape != shape or out_maps[cam].dtype != np.float16:
                    raise ValueError(f"wrong feature array contract: {path}")
        if not progress_path.exists():
            done = np.lib.format.open_memmap(progress_path, mode="w+", dtype=np.bool_, shape=(n,))
            done[:] = False
            done.flush()
        else:
            done = np.load(progress_path, mmap_mode="r+", allow_pickle=False)
        # Work row-batches with both cameras; atomically mark only after both
        # feature arrays for that row have been flushed.
        pending = np.flatnonzero(~np.asarray(done))
        for start in range(0, len(pending), int(batch_size)):
            ids = pending[start:start + int(batch_size)]
            for cam, now_name, prev_name in (
                ("base", "image_base", "image_base_prev"),
                ("wrist", "image_wrist", "image_wrist_prev"),
            ):
                src_ids = selected_rows[ids]
                clips = [_resize_clip(np.stack([imgs[prev_name][i], imgs[now_name][i]], axis=0),
                                      input_resolution) for i in src_ids]
                encoded = processor(videos=clips, return_tensors="pt")
                video = encoded["pixel_values_videos"].cuda().to(dtype=torch.float16)
                with torch.inference_mode():
                    feat = model.get_vision_features(pixel_values_videos=video)
                feat = feat.detach().to(device="cpu", dtype=torch.float16).numpy()
                if feat.ndim != 3 or feat.shape[0] != len(ids) or feat.shape[2] != 1024:
                    raise ValueError(f"unexpected {cam} encoder output {feat.shape}")
                if pool_tokens:
                    feat = pool_2x2_tokens(feat)
                if feat.shape[1:] != shape[1:]:
                    raise ValueError(f"unexpected pooled {cam} shape {feat.shape}")
                out_maps[cam][ids] = feat
            for cam in out_maps:
                out_maps[cam].flush()
            done[ids] = True
            done.flush()
            if start == 0 or (start // batch_size) % 100 == 0:
                (dst / "progress.json").write_text(json.dumps({
                    "split": split, "complete_rows": int(np.asarray(done).sum()),
                    "total_rows": n, "elapsed_seconds": time.time() - started}, indent=2) + "\n")
        if not np.asarray(done).all():
            raise RuntimeError(f"incomplete V-JEPA rows remain for {split}")
        # Persist exact row keys separately to make downstream joins auditable.
        key_path = dst / "row_keys.npy"
        if key_path.exists():
            existing = np.load(key_path, mmap_mode="r", allow_pickle=False)
            if not np.array_equal(existing, selected_keys):
                raise ValueError(f"row key cache differs: {key_path}")
        else:
            np.save(key_path, np.asarray(selected_keys, dtype=np.int32), allow_pickle=False)
        (dst / "manifest.json").write_text(json.dumps({
            "contract": "latent-probe-vjepa-v1", "split": split,
            "rows": n, "source_rows": n_total, "key_fields": ["task", "episode", "frame"],
            "row_key_sha256": key_hash, "row_keys": "row_keys.npy",
            "model_id": MODEL_ID, "model_dir": str(model_path),
            "model_config_sha256": _sha256(model_path / "config.json"),
            "dtype": "float16", "camera_outputs": {cam: f"{cam}.npy" for cam in out_maps},
            "feature_shape_per_camera": list(shape[1:]),
            "temporal_input": ["previous", "current"],
            "input_resolution": input_resolution,
            "input_resize_filter": "PIL BICUBIC" if input_resolution else "none",
            "cameras_processed_independently": True,
            "spatial_pool_2x2": bool(pool_tokens),
            "encoder_frozen": True,
            "target_fields_read": [], "complete": True,
        }, indent=2) + "\n")
        summaries[split] = {"rows": n, "shape_per_camera": list(shape[1:]),
                            "key_sha256": key_hash, "pooled": bool(pool_tokens)}
    root_manifest = {"contract": "latent-probe-vjepa-v1", "model_id": MODEL_ID,
                     "dtype": "float16", "splits": summaries, "complete": True,
                     "input_resolution": input_resolution,
                     "input_resize_filter": "PIL BICUBIC" if input_resolution else "none",
                     "elapsed_seconds": time.time() - started}
    tmp = output_root / "manifest.json.tmp"
    tmp.write_text(json.dumps(root_manifest, indent=2) + "\n")
    tmp.replace(output_root / "manifest.json")
    return root_manifest


def load_keyed_features(feature_root, split, expected_keys=None, *, mmap_mode="r"):
    """Load base/wrist arrays after verifying source and optional expected keys."""
    root = Path(feature_root) / split
    meta = json.loads((root / "manifest.json").read_text())
    if not meta.get("complete"):
        raise RuntimeError(f"feature split is incomplete: {root}")
    keys = np.load(root / meta["row_keys"], mmap_mode=mmap_mode, allow_pickle=False)
    if expected_keys is not None and not np.array_equal(np.asarray(expected_keys, np.int32), keys):
        raise ValueError("feature rows do not exactly match requested (task,episode,frame) keys")
    base = np.load(root / meta["camera_outputs"]["base"], mmap_mode=mmap_mode, allow_pickle=False)
    wrist = np.load(root / meta["camera_outputs"]["wrist"], mmap_mode=mmap_mode, allow_pickle=False)
    if len(keys) != len(base) or len(keys) != len(wrist):
        raise ValueError("feature/key row count mismatch")
    return {"keys": keys, "base": base, "wrist": wrist, "manifest": meta}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--input-resolution", type=int, choices=(128, 224, 256), default=None)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--memory-fraction", type=float, default=0.25)
    p.add_argument("--pool-2x2", action="store_true", help="store 64 pooled tokens/camera instead of 256")
    args = p.parse_args(argv)
    result = extract_vjepa(args.data_root, args.output_root, model_dir=args.model_dir,
        batch_size=args.batch_size, pool_tokens=args.pool_2x2, memory_fraction=args.memory_fraction,
        input_resolution=args.input_resolution)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
