"""Regenerate RES1 saved contexts at the exact unit-start observations.

Raw VLM tokens use ``vq_sources.Pi05Features`` (the vq-b-01 path); unlike the
old stride-5 lookup, every unit start is read directly from its LeRobot row.
Outputs go to a new directory and never overwrite the earlier diagnostic cache.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kq2_run as KR
import kq2cl_common as C
import res1_context as RC
import vq_sources as VS

OUT = C.ROOT / "res1-01" / "context-exact-01"


def overlap_ids(source: np.ndarray, cached_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return bank IDs and cache IDs for exact (task,episode,frame) matches."""
    lookup = {tuple(map(int, key)): i for i, key in enumerate(np.asarray(cached_keys))}
    pairs = [(u, lookup[tuple(map(int, row[:3]))]) for u, row in enumerate(source)
             if tuple(map(int, row[:3])) in lookup]
    if not pairs:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    a, b = zip(*pairs)
    return np.asarray(a, np.int64), np.asarray(b, np.int64)


def token_metrics(actual: np.ndarray, cached: np.ndarray) -> dict:
    a = np.asarray(actual, np.float32).reshape(-1, actual.shape[-1])
    b = np.asarray(cached, np.float32).reshape(a.shape)
    rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-5)
    cos = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-8)
    ap, bp = a.mean(1), b.mean(1)
    pc = (ap * bp).sum(-1) / np.maximum(np.linalg.norm(ap, axis=-1) * np.linalg.norm(bp, axis=-1), 1e-8)
    return {
        "values": int(a.size), "relative_error_median": float(np.median(rel)),
        "relative_error_p95": float(np.quantile(rel, .95)),
        "token_cosine_median": float(np.median(cos)), "token_cosine_min": float(cos.min()),
        "mean_pooled_cosine_median": float(np.median(pc)), "mean_pooled_cosine_min": float(pc.min()),
        "max_abs": float(np.max(np.abs(a - b))),
    }


def _write_npz_atomic(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def extract_all(output: Path = OUT, *, batch: int = 8, device: str = "cuda") -> dict:
    """Extract exact unit-start raw VLM tokens and both query-context variants."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "bank_vlm_raw.npz"
    report_path = output / "extraction.json"

    # KQ2's bank order and cached exact rows are the source of truth.
    ctx = KR.Ctx("cpu", splits=("eh",), vlm=True, out=C.KQ2)
    bank = ctx.bank
    source = np.asarray(bank["source"], np.int64)
    sp = ctx.splits["eh"]
    cache_keys = np.asarray(sp["key"], np.int32)
    old_cache = np.asarray(sp["vlm"], np.float16)
    if source.shape != (4516, 4) or len(cache_keys) != len(old_cache):
        raise AssertionError("unexpected KQ2 bank/cache dimensions")
    hit_u, hit_c = overlap_ids(source, cache_keys)
    if (len(hit_u), 4516 - len(hit_u)) != (1222, 3294):
        raise AssertionError(f"expected 1222 exact cached starts and 3294 missing, got {len(hit_u)}")

    if raw_path.exists():
        with np.load(raw_path, allow_pickle=False) as z:
            if not np.array_equal(z["source"], source):
                raise ValueError("existing exact VLM cache has different bank ordering")
            raw = z["vlm_raw"]
    else:
        if device.startswith("cuda"):
            torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(float(os.environ.get("TORCH_MEM_FRACTION", ".15")))
        reader = VS.ObsReader()
        fx = VS.Pi05Features()
        raw = np.lib.format.open_memmap(output / "bank_vlm_raw.npy", mode="w+", dtype=np.float16,
                                       shape=(len(source), 34, 2048))
        # Group reads by episode so parquet decoding is reused while preserving bank order.
        order = np.lexsort((source[:, 2], source[:, 1], source[:, 0]))
        for s in range(0, len(order), batch):
            ix = order[s:s + batch]
            items = [reader.read(source[j, :3], "libero40") for j in ix]
            feat = fx(items)
            vals = np.concatenate((feat["vlm_img"].astype(np.float16),
                                   feat["vlm_txt_mean"][:, None].astype(np.float16),
                                   feat["vlm_txt_last"][:, None].astype(np.float16)), axis=1)
            if vals.shape != (len(ix), 34, 2048):
                raise AssertionError(f"unexpected vq-b token shape {vals.shape}")
            raw[ix] = vals
        raw.flush()
        # Atomic compressed archive is the stable interchange artifact.
        _write_npz_atomic(raw_path, source=source, vlm_raw=np.asarray(raw))
        del raw
        with np.load(raw_path, allow_pickle=False) as z:
            raw = z["vlm_raw"]

    # Verify all 1,222 same-time rows against the existing KQ2/VQ-B feature cache.
    overlap = token_metrics(raw[hit_u], old_cache[hit_c])
    if overlap["token_cosine_median"] < .999 or overlap["mean_pooled_cosine_median"] < .999:
        raise RuntimeError(f"exact-start VLM cache parity failed: {overlap}")

    model_i, _ = C.load_query("i", torch.device(device))
    model_iii, _ = C.load_query("iii", torch.device(device))
    # For the no-VLM branch both representations use the exact same corrected
    # KQ2 episode-sequence indices as the checker-validated RES1-i cache.
    source_i = output / "bank_i.npz"
    source_iii = output / "bank_iii.npz"
    if source_i.exists() and source_iii.exists() and report_path.exists():
        return json.loads(report_path.read_text())

    gs = KR.GPUSplit(sp, torch.device(device), True)
    slot_by_ep = {}
    for row, key in enumerate(cache_keys):
        slot_by_ep.setdefault((int(key[0]), int(key[1])), int(gs.slot[row]))
    slots = np.asarray([slot_by_ep[(int(t), int(e))] for t, e, _, _ in source], np.int64)

    class UnitSplit:
        def __init__(self, use_vlm: bool):
            self.seq_a, self.seq_s, self.off = gs.seq_a, gs.seq_s, gs.off
            self.slot = torch.as_tensor(slots, dtype=torch.long, device=torch.device(device))
            self.start = torch.as_tensor(source[:, 2], dtype=torch.long, device=torch.device(device))
            self.vlm = None
            if use_vlm:
                # query_context applies the KQ2 QueryNet.kv projection itself.
                self.vlm = torch.as_tensor(np.asarray(raw), dtype=torch.float16, device=device)

        def window(self, ix, length=10):
            t = self.start[ix][:, None]
            g = self.off[self.slot[ix]][:, None]
            rel = torch.arange(length, device=device)[None] - (length - 1)
            tau = t + rel
            valid = tau >= 0
            gi = (g + t + rel).clamp(min=0)
            states = self.seq_s[gi] * valid[..., None]
            av = tau >= 1
            actions = self.seq_a[(g + t + rel - 1).clamp(min=0)] * av[..., None]
            return torch.cat((actions, states, av[..., None].float()), -1), valid

    h_by = {}
    state_by = {}
    vlm_projected = {}
    for which, model in (("i", model_i), ("iii", model_iii)):
        us = UnitSplit(which == "iii")
        hs, sts, vls = [], [], []
        model.eval()
        with torch.inference_mode():
            for lo in range(0, len(source), batch):
                ids = torch.arange(lo, min(lo + batch, len(source)), device=device)
                state = gs.seq_s[gs.off[us.slot[ids]] + us.start[ids]]
                qctx = RC.query_context(model, us, ids, state=state, visual=(which == "iii"))
                hs.append(qctx["h"].cpu().numpy().astype(np.float32))
                sts.append(qctx["state"].cpu().numpy().astype(np.float32))
                if which == "iii":
                    vls.append(qctx["vlm"].cpu().numpy().astype(np.float16))
        h_by[which] = np.concatenate(hs)
        state_by[which] = np.concatenate(sts)
        if which == "iii":
            vlm_projected[which] = np.concatenate(vls)
        p = output / f"bank_{which}.npz"
        payload = {"h": h_by[which], "state": state_by[which],
                   "bank_task": np.asarray(bank["task"], np.int32),
                   "bank_episode": np.asarray(bank["episode"], np.int32), "bank_source": source}
        if which == "iii":
            payload["vlm"] = vlm_projected[which]
            payload["vlm_valid"] = np.ones((len(source), 34), bool)
        _write_npz_atomic(p, **payload)
        (output / f"bank_{which}.json").write_text(json.dumps({
            "context_version": 3, "which": which, "rows": len(source), "bank_source": "KQ2 K-x1 unit bank source order",
            "unit_start_exact": True, "vlm_start_exact": which == "iii",
            "query_checkpoint": str(C.QUERY_CK[which]),
            "history_state": "KQ2 exact sequence index at bank source start frame",
            "vlm": "vq_sources Pi05Features at the exact LeRobot start frame, then KQ2 QueryNet.kv" if which == "iii" else None,
            "contains_target_task_id_or_future": False,
        }, indent=2) + "\n")

    report = {
        "rows": int(len(source)), "exact_overlap_rows": int(len(hit_u)), "newly_extracted_rows": int(4516 - len(hit_u)),
        "feature_path": "vq_sources.ObsReader.read(libero40) -> vq_sources.Pi05Features (same pi05_libero model, corrected LeRobot orientation, 32 image pools + prompt mean/last)",
        "cache_comparison": overlap,
        "source_order_matches_bank": True,
        "unit_start_exact": True,
        "context_shapes": {"i_h": list(h_by["i"].shape), "iii_h": list(h_by["iii"].shape),
                           "iii_vlm": list(vlm_projected["iii"].shape)},
        "contexts": {"i": str(source_i), "iii": str(source_iii)},
        "no_previous_frame_substitution": True,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    print(json.dumps(extract_all(a.output, batch=a.batch, device=a.device), indent=2))


if __name__ == "__main__":
    main()
