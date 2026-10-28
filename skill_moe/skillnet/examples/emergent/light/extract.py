"""LeRobot LIBERO frame loader and frozen pi0.5 PaliGemma feature extractor."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

SUITE_BY_TASK = {**{i: "libero_goal" for i in range(10, 20)},
                 **{i: "libero_object" for i in range(20, 30)},
                 **{i: "libero_spatial" for i in range(30, 40)}}
READ_COLUMNS = ["image", "wrist_image", "state", "actions", "episode_index", "task_index", "frame_index"]


def _image_bytes(column, i: int) -> bytes:
    value = column[i].as_py()
    return value["bytes"] if isinstance(value, dict) else value


def load_episode(path: Path, episode_meta: dict[str, Any]) -> dict[str, Any] | None:
    """Read only model/data columns; task annotations and skill columns are never selected."""
    table = pq.read_table(path, columns=READ_COLUMNS)
    task_ids = np.asarray(table["task_index"].to_numpy())
    task_id = int(task_ids[0])
    if task_id not in SUITE_BY_TASK:
        return None
    steps = np.asarray(table["frame_index"].to_numpy(), dtype=np.int32)
    state = np.asarray(table["state"].combine_chunks().values.to_numpy(), dtype=np.float32)
    action = np.asarray(table["actions"].combine_chunks().values.to_numpy(), dtype=np.float32)
    state = state.reshape(len(steps), -1)
    action = action.reshape(len(steps), -1)
    return {"episode_id": f"libero40-{int(episode_meta['episode_index']):06d}",
            "task_index": task_id, "suite": SUITE_BY_TASK[task_id],
            "task": str(episode_meta["tasks"][0]), "steps": steps,
            "state": state, "actions": action,
            "images": [_image_bytes(table["image"], i) for i in range(len(steps))],
            "wrist_images": [_image_bytes(table["wrist_image"], i) for i in range(len(steps))]}


def dataset_episodes(dataset_dir: Path):
    metas = {int(d["episode_index"]): d for d in
             (json.loads(line) for line in (dataset_dir / "meta/episodes.jsonl").open())}
    for p in sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet")):
        idx = int(p.stem.split("_")[-1])
        ep = load_episode(p, metas[idx])
        if ep is not None:
            yield ep


def _pooler(checkpoint: Path, batch_size: int):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models import pi0_config
    from openpi.models.model import restore_params
    from openpi.shared import image_tools

    cfg = pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m",
                               action_horizon=25, max_token_len=200, dtype="bfloat16")
    model = cfg.load(restore_params(checkpoint, dtype=jnp.bfloat16))
    resize = jax.jit(lambda x: image_tools.resize_with_pad(x, 224, 224))

    def infer(images, tokens, token_mask, has_language):
        image_tokens = []
        for image in images:
            tok, _ = model.PaliGemma.img(image, train=False)
            image_tokens.append(tok)
        prefix = jnp.concatenate(image_tokens, axis=1)
        masks = [jnp.ones((prefix.shape[0], prefix.shape[1]), dtype=jnp.bool_)]
        ar = [jnp.zeros((prefix.shape[1],), dtype=jnp.bool_)]
        if has_language:
            text = model.PaliGemma.llm(tokens, method="embed")
            prefix = jnp.concatenate([prefix, text], axis=1)
            masks.append(token_mask)
            ar.append(jnp.zeros((text.shape[1],), dtype=jnp.bool_))
        input_mask = jnp.concatenate(masks, axis=1)
        ar_mask = jnp.concatenate(ar)
        attention = (jnp.cumsum(ar_mask)[None, :, None] <= jnp.cumsum(ar_mask)[None, None, :])
        attention = attention & input_mask[:, None, :] & input_mask[:, :, None]
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (out, _), _ = model.PaliGemma.llm([prefix, None], mask=attention, positions=positions,
                                          adarms_cond=[None, None])
        return jnp.mean(out[:, :512, :].astype(jnp.float32), axis=1)

    infer_jit = jax.jit(infer, static_argnums=(3,))
    from openpi.models.tokenizer import PaligemmaTokenizer
    tokenizer = PaligemmaTokenizer(200)

    def extract(image_bytes, wrist_bytes, tasks):
        batch = []
        for a, b in zip(image_bytes, wrist_bytes):
            aa = np.asarray(Image.open(io.BytesIO(a)).convert("RGB"), dtype=np.uint8)
            bb = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)
            batch.append(np.stack([aa, bb]))
        image = np.stack(batch) # B,2,H,W,C
        flat = image.reshape(-1, *image.shape[2:])
        padded = np.asarray(resize(jnp.asarray(flat)))
        padded = jnp.asarray(padded, dtype=jnp.float32) / 127.5 - 1.0
        images = [padded.reshape(len(batch), 2, 224, 224, 3)[:, i] for i in range(2)]
        toks, masks = zip(*(tokenizer.tokenize(task, None) for task in tasks))
        tokens, token_mask = jnp.asarray(np.stack(toks)), jnp.asarray(np.stack(masks))
        lang = infer_jit(images, tokens, token_mask, True)
        nolang = infer_jit(images, tokens, token_mask, False)
        return np.asarray(lang, dtype=np.float32), np.asarray(nolang, dtype=np.float32)
    return extract


def extract_cache(dataset_dir: Path, cache_dir: Path, checkpoint: Path, *, stride: int = 5,
                  batch_size: int = 2, max_episodes: int | None = None) -> dict[str, Any]:
    """Extract cached f/k pools at stride; resume by skipping complete episode cache files."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ep_dir = cache_dir / "episodes"
    ep_dir.mkdir(exist_ok=True)
    target = cache_dir / "manifest.json"
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        expected = (str(dataset_dir), str(checkpoint), stride)
        actual = (previous.get("dataset"), previous.get("checkpoint"), previous.get("stride"))
        if actual != expected:
            raise ValueError(f"feature cache manifest mismatch: expected {expected}, found {actual}")
        if max_episodes is not None:
            previous = {**previous, "episodes": previous["episodes"][:max_episodes]}
        return previous
    extract = _pooler(checkpoint, batch_size)
    inventory = []
    started = time.monotonic()
    seen = 0
    for n, ep in enumerate(dataset_episodes(dataset_dir)):
        if max_episodes is not None and n >= max_episodes:
            break
        path = ep_dir / f"{ep['episode_id']}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as d:
                inventory.append({"episode_id": ep["episode_id"], "suite": ep["suite"], "task": ep["task"],
                                  "task_index": ep["task_index"], "n_rows": len(d["step"]),
                                  "n_frames": len(ep["steps"]), "cache": str(path)})
            continue
        indices = np.arange(0, len(ep["steps"]), stride, dtype=np.int32)
        feats_l, feats_k = [], []
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            # static batch shapes avoid repeated compilation; replicate final sample if needed.
            real_n = len(idx)
            if real_n < batch_size:
                idx = np.pad(idx, (0, batch_size - real_n), mode="edge")
            fl, fk = extract([ep["images"][i] for i in idx], [ep["wrist_images"][i] for i in idx],
                             [ep["task"]] * len(idx))
            feats_l.extend(fl[:real_n]); feats_k.extend(fk[:real_n])
        # Environment state convention: xyz, axis-angle xyz, two gripper qpos.
        payload = {"step": ep["steps"][indices], "env_step": ep["steps"], "proprio": ep["state"][indices],
                   "action": ep["actions"], "env_state": ep["state"],
                   "f": np.asarray(feats_l, np.float16), "k": np.asarray(feats_k, np.float16),
                   "task_index": np.asarray(ep["task_index"], np.int32)}
        with path.open("xb") as f:
            np.savez_compressed(f, **payload)
        inventory.append({"episode_id": ep["episode_id"], "suite": ep["suite"], "task": ep["task"],
                          "task_index": ep["task_index"], "n_rows": len(indices), "n_frames": len(ep["steps"]),
                          "cache": str(path)})
        seen += 1
        if seen % 10 == 0 or seen == 1:
            elapsed = time.monotonic() - started
            print(f"feature extraction: episodes={seen} sampled_frames={sum(x['n_rows'] for x in inventory)} "
                  f"elapsed_min={elapsed / 60:.1f}", flush=True)
        # release decoded image buffers between episodes
        del ep
    manifest = {"dataset": str(dataset_dir), "checkpoint": str(checkpoint), "stride": stride,
                "vision": "pi0.5 base PaliGemma; base+wrist; mean image tokens; float16 cache",
                "episodes": inventory}
    with target.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
