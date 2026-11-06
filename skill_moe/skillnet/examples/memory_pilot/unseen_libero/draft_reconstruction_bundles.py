#!/usr/bin/env python3
"""Self encode/decode learned shape-skill bundles on Long and unseen rows.

No retrieval or Expert is used.  Every evaluation segment is encoded from its
own action trace with a frozen learned bundle, decoded, then sampled at the
same 10-frame windows/phase offsets used by shape_skills.run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent / "pi05_memory"
SHAPE = PI05 / "shape_skills"
REPO_ROOT = Path(__file__).resolve().parents[5]
STATE_DIR = Path(os.environ.get("SKILLNET_STATE_DIR", REPO_ROOT / ".skillnet-state")).expanduser()
PI05_ROOT = Path(os.environ.get("SKILLNET_PI05_ROOT", STATE_DIR / "outputs/memory-pilot/pi05-memory40-20261104")).expanduser()
ROOT = STATE_DIR / "outputs/memory-pilot"
RUN = PI05_ROOT / "run-01"
UNSEEN = Path(os.environ.get("UNSEEN_LIBERO_OUT", ROOT / "unseen-libero/oracle-01")).expanduser()
TOPK = PI05_ROOT / "topk-long-01/production/predictions_and_errors.npz"
BUNDLES = PI05_ROOT / "shape-skills-02/corrected-v5"
CHECKPOINT = Path(os.environ.get("SKILLNET_PI05_CHECKPOINT", STATE_DIR / "checkpoints/openpi-assets/checkpoints/pi05_libero")).expanduser()
TARGET = ROOT / "unseen-libero/draft-reconstruction-01/bundles.json"

sys.path.insert(0, str(SHAPE))
import data as shape_data  # noqa: E402
import run as shape_run  # noqa: E402

GROUPS = {"xyz": (0, 3), "rot": (3, 6), "grip": (6, 7)}
SEGMENTATION_ORDER = (
    "S0_fixed10", "S1_greedy_e0p01", "S1_greedy_e0p04",
    "S2_grip", "S2_grip_velocity", "S3_type_change",
)


def summarize(pred, target, hard):
    """Keep metric definitions byte-for-byte equivalent to the original run."""
    p, t = np.asarray(pred)[:, :, :7], np.asarray(target)[:, :, :7]
    e = np.mean((p - t) ** 2, axis=(1, 2))
    u, v = p[:, :, :3].sum(1), t[:, :, :3].sum(1)
    n = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    cos = np.divide((u * v).sum(1), n, out=np.ones(len(u)), where=n > 1e-8)
    out = {"all": float(e.mean()), "hard": float(e[hard].mean()),
           "median": float(np.median(e)), "p95": float(np.quantile(e, .95)),
           "within_01": float((e <= .01).mean()), "within_04": float((e <= .04).mean()),
           "xyz_cos": float(cos.mean()), "hard_xyz_cos": float(cos[hard].mean()),
           "grip_sign": float(np.mean(np.sign(p[:, :, 6]) == np.sign(t[:, :, 6])))}
    for name, (lo, hi) in GROUPS.items():
        out[f"mse_{name}"] = float(np.mean((p[:, :, lo:hi] - t[:, :, lo:hi]) ** 2))
    return out


def interp_window(actions, start, n=10):
    """Same phase rule as shape_skills.run.interp_window."""
    a = np.asarray(actions, np.float32)
    if a.ndim != 2 or not len(a):
        raise ValueError("decoded segment must be a nonempty [time,channels] array")
    t = np.clip(np.arange(n, dtype=np.float32) + float(start), 0, len(a) - 1)
    i = np.floor(t).astype(np.int64)
    j = np.minimum(i + 1, len(a) - 1)
    u = t - i
    return a[i] * (1 - u[:, None]) + a[j] * u[:, None]


def align_by_row_ids(expected_ids, result_ids, values):
    """Restore canonical query order without confusing row ids with pool offsets."""
    expected_ids = np.asarray(expected_ids, np.int64)
    result_ids = np.asarray(result_ids, np.int64)
    values = np.asarray(values)
    if len(np.unique(result_ids)) != len(result_ids) or set(expected_ids.tolist()) != set(result_ids.tolist()):
        raise AssertionError("result row IDs do not exactly cover the requested dataset row IDs")
    pos = {int(row_id): i for i, row_id in enumerate(result_ids)}
    return values[np.asarray([pos[int(row_id)] for row_id in expected_ids], np.int64)]


def roundtrip_segment(actions, encode, decode):
    """Apply one frozen segment encoder/decoder and enforce length/shape parity."""
    source = np.asarray(actions, np.float32)
    code, pose = encode(source)
    reconstructed = np.asarray(decode(code, pose, len(source)), np.float32)
    if reconstructed.shape != source.shape:
        raise AssertionError(f"segment round-trip shape {reconstructed.shape} != {source.shape}")
    return reconstructed


def _unseen_episodes(un, q01, q99):
    """Stitch cached stride-10 action windows into episode-local frame traces."""
    task = np.asarray(un["task"], np.int32)
    episode = np.asarray(un["episode"], np.int32)
    frame = np.asarray(un["frame"], np.int32)
    actions = np.asarray(un["actions"], np.float32)
    episodes = {}
    for key in sorted(set(zip(task.tolist(), episode.tolist()))):
        ix = np.flatnonzero((task == key[0]) & (episode == key[1]))
        starts = frame[ix]
        length = int(np.max(starts + actions.shape[1]))
        norm = np.full((length, 7), np.nan, np.float32)
        for row, start in zip(ix, starts, strict=True):
            chunk = actions[row, :, :7]
            end = int(start) + len(chunk)
            overlap = np.isfinite(norm[start:end]).all(1)
            if np.any(overlap):
                np.testing.assert_allclose(norm[start:end][overlap], chunk[overlap], atol=2e-6, rtol=0)
            norm[start:end] = chunk
        # Traces are only required through complete cached windows; an omitted
        # terminal remainder is never queried. Reject interior gaps explicitly.
        finite_rows = np.isfinite(norm).all(1)
        last = int(max(starts) + actions.shape[1])
        if not np.all(finite_rows[:last]):
            raise AssertionError(f"gapped cached episode {key}")
        norm = norm[:last]
        phys = shape_data.unnormalize(norm, q01[:7], q99[:7])
        episodes[key] = {"task": int(key[0]), "episode": int(key[1]),
                         "frame": np.arange(len(norm), dtype=np.int32),
                         "actions_norm": norm, "actions_phys": phys,
                         "state_phys": None,
                         "meta": {"memory": False, "validation": False, "present": True}}
    return episodes


def _requested_source_segments(episodes, rows):
    """Rebuild only requested segmentations with run.py's Dev thresholds."""
    dev_keys = [k for k, v in episodes.items()
                if k[0] in shape_data.HELDOUT_TASKS and v["meta"]["validation"]]
    velocity, velocity_record = shape_data.choose_velocity_threshold(episodes, dev_keys)
    type_threshold, type_record = shape_data.choose_type_threshold(episodes, dev_keys)
    specs = {
        "S0_fixed10": ("fixed10", None),
        "S1_greedy_e0p01": ("greedy", .01),
        "S1_greedy_e0p04": ("greedy", .04),
        "S2_grip": ("grip", None),
        "S2_grip_velocity": ("gripvel", velocity),
        "S3_type_change": ("typechange", type_threshold),
    }
    lookup = shape_run.make_row_lookup(rows)
    result = {}
    for name, (kind, param) in specs.items():
        by_ep = {}
        for key, seq in episodes.items():
            if kind == "fixed10": bounds = shape_data.fixed10(seq)
            elif kind == "greedy": bounds = shape_data.threshold_segments(seq, param, "greedy")
            elif kind == "grip": bounds = shape_data.grip_segments(seq)
            elif kind == "gripvel": bounds = shape_data.grip_velocity_segments(seq, param)
            else: bounds = shape_data.type_change_segments(seq, param)
            for start, end in bounds:
                if end - start < 10:
                    continue
                frame = int(seq["frame"][start])
                seg = {"task": int(key[0]), "episode": int(key[1]), "start": frame,
                       "length": int(end-start), "row_id": int(lookup.get((key[0], key[1], frame), -1)),
                       "a": seq["actions_phys"][start:end], "meta": seq["meta"]}
                by_ep.setdefault(key, []).append(seg)
        for key in by_ep:
            by_ep[key].sort(key=lambda s: s["start"])
        result[name] = by_ep
    return result, {"velocity": velocity_record, "type_change": type_record}


def _vq_procrustes_references(segsets):
    """Recreate VQ's frozen Train-only phase reference from its source rule."""
    import representations as shape_repr
    refs = {}
    excluded = set(shape_data.LONG_TASKS) | set(shape_data.HELDOUT_TASKS)
    for name, by_ep in segsets.items():
        paths = []
        for segs in by_ep.values():
            for seg in segs:
                if int(seg["task"]) in excluded:
                    continue
                can, _ = shape_repr.canonicalize(np.asarray(seg["a"], np.float32)[:, :7])
                path = shape_repr.resample(np.cumsum(can[:, :3], axis=0), shape_repr.PHASE)
                paths.append(path - path.mean(0, keepdims=True))
        if not paths:
            raise ValueError(f"no VQ fit segments for {name}")
        ref = np.mean(paths, axis=0).astype(np.float32)
        refs[name] = ref - ref.mean(0, keepdims=True)
    return refs


def _ae_procrustes_references(segsets):
    """Recreate AE's Train-episode-only Procrustes reference if not bundled."""
    import representations as shape_repr
    refs = {}
    excluded = set(shape_data.LONG_TASKS) | set(shape_data.HELDOUT_TASKS)
    for name, by_ep in segsets.items():
        paths = []
        for segs in by_ep.values():
            for seg in segs:
                meta = seg.get("meta", {})
                if int(seg["task"]) in excluded or not meta.get("present", False):
                    continue
                if meta.get("memory", False) or meta.get("validation", False):
                    continue
                can, _ = shape_repr.canonicalize(np.asarray(seg["a"], np.float32)[:, :7])
                path = shape_repr.resample(np.cumsum(can[:, :3], axis=0), shape_repr.PHASE)
                paths.append(path - path.mean(0, keepdims=True))
        if not paths:
            raise ValueError(f"no AE train episodes for {name}")
        ref = np.mean(paths, axis=0).astype(np.float32)
        refs[name] = ref - ref.mean(0, keepdims=True)
    return refs


def _segment_unseen(episodes, rows, thresholds, seg_name):
    row_lookup = shape_run.make_row_lookup(rows)
    if seg_name == "S0_fixed10":
        kind, param = "fixed10", None
    elif seg_name.startswith("S1_greedy_"):
        kind, param = "greedy", float(seg_name.split("e")[-1].replace("p", "."))
    elif seg_name == "S2_grip":
        kind, param = "grip", None
    elif seg_name == "S2_grip_velocity":
        kind, param = "gripvel", float(thresholds["velocity"]["selected_threshold"])
    elif seg_name == "S3_type_change":
        kind, param = "typechange", float(thresholds["type_change"]["selected_threshold"])
    else:
        raise ValueError(f"unsupported segmentation for bundle extension: {seg_name}")
    by_ep = {}
    for key, seq in episodes.items():
        if kind == "fixed10": bounds = shape_data.fixed10(seq)
        elif kind == "greedy": bounds = shape_data.threshold_segments(seq, param, "greedy")
        elif kind == "grip": bounds = shape_data.grip_segments(seq)
        elif kind == "gripvel": bounds = shape_data.grip_velocity_segments(seq, param)
        else: bounds = shape_data.type_change_segments(seq, param)
        items = []
        for start, end in bounds:
            if end - start < 10:
                continue
            frame = int(seq["frame"][start])
            items.append({"task": key[0], "episode": key[1], "start": frame,
                          "length": int(end - start),
                          "row_id": int(row_lookup.get((key[0], key[1], frame), -1)),
                          "a": seq["actions_phys"][start:end]})
        by_ep[key] = items
    return by_ep


def _make_queries_for_unseen(seg_by_ep, row_ids, un):
    row_task = np.asarray(un["task"])
    row_ep = np.asarray(un["episode"])
    row_frame = np.asarray(un["frame"])
    row_actions = np.asarray(un["actions"])
    result = {}
    for rid in row_ids:
        key = (int(row_task[rid]), int(row_ep[rid]))
        segs = seg_by_ep[key]
        starts = np.asarray([s["start"] for s in segs], np.int64)
        loc = int(np.searchsorted(starts, int(row_frame[rid]), side="right") - 1)
        if loc < 0 or int(row_frame[rid]) >= segs[loc]["start"] + segs[loc]["length"]:
            start = int(row_frame[rid])
            segs.append({"task": key[0], "episode": key[1], "start": start,
                         "length": 10, "row_id": int(rid),
                         "a": shape_data.unnormalize(row_actions[rid, :, :7],
                                                     np.asarray(un["q01"]), np.asarray(un["q99"]))})
            segs.sort(key=lambda s: s["start"])
            starts = np.asarray([s["start"] for s in segs], np.int64)
            loc = int(np.searchsorted(starts, start, side="right") - 1)
        result.setdefault(key, {})[int(rid)] = (segs[loc], float(row_frame[rid] - segs[loc]["start"]))
    return result


def _long_queries(segsets, row_ids, rows, episodes):
    row_task = np.asarray(rows["task"])
    row_ep = np.asarray(rows["episode"])
    row_frame = np.asarray(rows["frame"])
    result = {}
    fallback_rows = set()
    for rid in row_ids:
        key = (int(row_task[rid]), int(row_ep[rid]))
        segs = segsets[key]
        starts = np.asarray([s["start"] for s in segs], np.int64)
        loc = int(np.searchsorted(starts, int(row_frame[rid]), side="right") - 1)
        if loc < 0 or int(row_frame[rid]) >= segs[loc]["start"] + segs[loc]["length"]:
            frame = int(row_frame[rid])
            seq = episodes[key]
            if frame + 10 > len(seq["frame"]):
                raise AssertionError(f"Long row {rid} has no complete action window")
            segs.append({"task": key[0], "episode": key[1], "start": frame,
                         "length": 10, "row_id": int(rid),
                         "a": seq["actions_phys"][frame:frame + 10], "meta": seq["meta"]})
            segs.sort(key=lambda s: s["start"])
            starts = np.asarray([s["start"] for s in segs], np.int64)
            loc = int(np.searchsorted(starts, frame, side="right") - 1)
            fallback_rows.add(int(rid))
        result.setdefault(key, {})[int(rid)] = (segs[loc], float(row_frame[rid] - segs[loc]["start"]))
    return result, fallback_rows


def _pose_values(pose_npz):
    n = len(pose_npz["length"])
    counts = np.zeros(n, np.float64)
    for name in pose_npz.files:
        arr = np.asarray(pose_npz[name])
        if arr.ndim == 0:
            counts += arr.size / max(n, 1)
        elif arr.shape[0] == n:
            counts += np.prod(arr.shape[1:], dtype=np.int64) if arr.ndim > 1 else 1
    return counts


def _eval_config(config_dir, seg_name, set_name, query_ids, hard, target, rows,
                 pooled, row_lookup, source_episodes, source_segsets,
                 query_episodes, query_segsets, vq_references, ae_references,
                 unseen=None, direct_model=None):
    out_root = config_dir.parent.parent
    config = config_dir.name
    if direct_model is None:
        model, _, _, _, _ = shape_run.load_eval_bundle(
            out_root, config, source_segsets[seg_name], source_episodes)
    else:
        model = direct_model
    # load_eval_bundle also exposes serialized codes for pool segments. This
    # diagnostic is strictly self encode->decode, so never use that shortcut
    # for either Long or unseen evaluation traces.
    model["codes_by_key"] = {}
    if model["kind"] == "vq" and model["input_mode"] == "procrustes":
        model["reference"] = vq_references[seg_name]
    if model["kind"] == "ae" and model["variant"] != "raw" and model.get("reference") is None:
        model["reference"] = ae_references[seg_name]
    if model["kind"] == "m1":
        raise AssertionError("M1 is already included in the original comparison; unexpected M1 bundle selected")
    if set_name == "long":
        queries, fallback_rows = _long_queries(query_segsets[seg_name], query_ids, rows, query_episodes)
        row_task, row_ep, row_frame = rows["task"], rows["episode"], rows["frame"]
    else:
        queries = _make_queries_for_unseen(query_segsets[seg_name], query_ids, unseen)
        fallback_rows = set()
        row_task, row_ep, row_frame = unseen["task"], unseen["episode"], unseen["frame"]
    # Cache one encode/decode per selected evaluation segment; all fixed-window
    # queries in a segment use the same decoded segment and continuous phase.
    decoded = {}
    predictions = np.empty((len(query_ids), 10, 7), np.float32)
    for pos, rid in enumerate(np.asarray(query_ids, np.int64)):
        key = (int(row_task[rid]), int(row_ep[rid]))
        seg, offset = queries[key][int(rid)]
        segkey = (key, int(seg["start"]), int(seg["length"]))
        if segkey not in decoded:
            code, pose = shape_run.encode_segment(model, seg, row_lookup, rows, pooled)
            manifest = json.loads((config_dir / "manifest.json").read_text())
            q01 = np.asarray(manifest["q01"], np.float32)
            q99 = np.asarray(manifest["q99"], np.float32)
            def encode_fn(action):
                local_seg = dict(seg, a=action)
                return shape_run.encode_segment(model, local_seg, row_lookup, rows, pooled)
            def decode_fn(local_code, local_pose, length):
                return shape_run.decoded_candidate(
                    model, local_code, local_pose, length, "raw", local_pose, q01, q99)
            physical = roundtrip_segment(seg["a"], encode_fn, decode_fn)
            decoded[segkey] = physical
        window_phys = shape_run.interp_window(decoded[segkey], offset, 10)
        predictions[pos] = shape_data.normalize_physical(window_phys, q01, q99)
    # Keep the exact query order asserted by the cached targets/hard mask.
    assert len(predictions) == len(target) == len(query_ids) and len(hard) == len(query_ids)
    summary = summarize(predictions, target, hard)
    seg_file = np.load(config_dir / "segments.npz", allow_pickle=False)
    pose_file = np.load(config_dir / "pose.npz", allow_pickle=False)
    lens = np.asarray(seg_file["length"], np.float64)
    code_dim = int(json.loads((config_dir / "manifest.json").read_text())["code_dim"])
    per_step = (code_dim + _pose_values(pose_file)) / np.maximum(lens, 1)
    return {"segmentation": seg_name, "family": model["kind"], "code_dim": code_dim,
            "stored_values_per_step_mean": float(per_step.mean()),
            "stored_values_per_step_median": float(np.median(per_step)),
            "query_segments_encoded": len(decoded), "query_only_fallback_rows": len(fallback_rows),
            "rows": len(query_ids),
            "hard_rows": int(np.asarray(hard).sum()), "metrics": summary,
            "predictions": predictions}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=TARGET)
    ap.add_argument("--bundle-root", type=Path, default=BUNDLES)
    ap.add_argument("--max-bundles", type=int, default=0,
                    help="debug-only limit, sorted bundle list; 0 evaluates all eligible bundles")
    ap.add_argument("--only", default="",
                    help="debug-only exact representation name")
    ap.add_argument("--vq1024-only", action="store_true",
                    help="evaluate available K=1024 VQ source bundles as a separate addendum")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    episodes, rows, source_meta = shape_data.load_source(RUN, CHECKPOINT)
    pooled = np.load(RUN / "pooled.npy", mmap_mode="r", allow_pickle=False)
    segsets_by_name, thresholds = _requested_source_segments(episodes, rows)
    vq_references = _vq_procrustes_references(segsets_by_name)
    ae_references = _ae_procrustes_references(segsets_by_name)
    segsets_list = {name: [s for segs in by_ep.values() for s in segs]
                    for name, by_ep in segsets_by_name.items()}

    topk = np.load(TOPK, allow_pickle=False)
    long_ids = np.asarray(topk["query_rows"], np.int64)
    long_mask = np.asarray(rows["task"])[long_ids] < 10
    long_ids = long_ids[long_mask]
    hard_long = np.asarray(topk["hard_mask"], bool)[long_mask]
    if len(long_ids) != 548 or int(hard_long.sum()) != 63:
        raise AssertionError("Long query selection differs from draft_reconstruction.py")

    unroot = UNSEEN / "unseen_rows"
    un = {p.stem: np.load(p, mmap_mode="r", allow_pickle=False)
          for p in unroot.glob("*.npy")}
    dev = np.asarray(un["is_dev"], bool)
    unseen_ids = np.flatnonzero(~dev).astype(np.int64)
    drafts = np.load(UNSEEN / "drafts/raw_none_eval.npz", allow_pickle=False)
    expected_unseen_qids = unseen_ids + len(rows["task"])
    if not np.array_equal(drafts["query_ids"], expected_unseen_qids):
        raise AssertionError("unseen eval rows differ from the original reconstruction cache")
    threshold = json.loads((UNSEEN / "hard_threshold.json").read_text())["threshold"]
    hard_unseen = np.asarray(drafts["pre_mse"] >= threshold, bool)
    if len(unseen_ids) != 619 or int(hard_unseen.sum()) != 134:
        raise AssertionError("unseen query/hard selection differs from draft_reconstruction.py")
    qstats = source_meta
    un = dict(un)
    un["q01"] = np.asarray(qstats["q01"], np.float32)[:7]
    un["q99"] = np.asarray(qstats["q99"], np.float32)[:7]
    un_episodes = _unseen_episodes(un, un["q01"], un["q99"])

    # Build only the segmentation families requested for unseen rows, using the
    # same frozen Dev-selected thresholds as the learned bundle family.
    unseen_segsets = {name: _segment_unseen(un_episodes, un, thresholds, name)
                      for name in SEGMENTATION_ORDER}
    # Evaluation target arrays and order match the original result script.
    long_target = np.asarray(rows["actions"][long_ids, :, :7], np.float32)
    unseen_target = np.asarray(un["actions"][unseen_ids, :, :7], np.float32)
    row_lookup = shape_run.make_row_lookup(rows)

    eligible = []
    if args.vq1024_only:
        if args.output == TARGET:
            raise ValueError("--vq1024-only requires a separate --output path; bundles.json is left intact")
        import vq as vqmod
        for source in sorted((args.bundle_root / "representations").glob("vq_*_K1024")):
            for source_r in sorted(source.glob("r*")):
                mf = source_r / "manifest.json"
                if not mf.exists():
                    continue
                meta = json.loads(mf.read_text())
                seg = meta.get("segment_name")
                if seg not in SEGMENTATION_ORDER:
                    continue
                net, net_meta, extras = vqmod._load_model_npz(source_r / "model.npz", device="cpu")
                mode = str(meta["input_mode"])
                model = {"kind": "vq", "input_mode": mode,
                         "residual_dim": int(meta["residual_dim"]),
                         "code_dim": int(meta["code_dim"]), "bundle_src": source_r,
                         "vq_model": net,
                         "stats": {"mean": np.asarray(extras["feature_mean"], np.float32),
                                   "std": np.asarray(extras["feature_std"], np.float32)},
                         "q01": np.asarray(meta["q01"], np.float32),
                         "q99": np.asarray(meta["q99"], np.float32),
                         "codes_by_key": {}, "reference": vq_references[seg] if mode == "procrustes" else None}
                label = f"{source.name}__{source_r.name}"
                eligible.append((label, source_r, seg, model))
    else:
        for d in sorted((args.bundle_root / "representations").glob("*")):
            mf = d / "manifest.json"
            if not d.is_dir() or not mf.exists():
                continue
            meta = json.loads(mf.read_text())
            seg = meta.get("segmentation")
            kind = meta.get("kind")
            if seg not in SEGMENTATION_ORDER or kind == "m1":
                continue
            # Existing comparisons already contain raw PCA and M1. Add only the
            # learned bundle families requested for this extension.
            name = d.name
            keep = (kind in {"spline", "dmp", "vq", "ae"} or
                    (kind in {"pca_canonical", "procrustes_pca", "pca_gripsep", "pca_raw_white"}))
            if not keep:
                continue
            if seg not in segsets_by_name:
                raise AssertionError(f"missing run-01 segmentation {seg}")
            eligible.append((name, d, seg, None))
    eligible.sort(key=lambda x: x[0])
    if args.only:
        eligible = [item for item in eligible if item[0] == args.only]
    if args.max_bundles:
        eligible = eligible[:args.max_bundles]
    if not eligible:
        raise FileNotFoundError("no eligible learned representation bundles")

    result = {"source": str(args.bundle_root),
              "method": "self encode/decode each query segment; no retrieval or Expert",
              "metric_definition": "copied summarize: mean squared error over each normalized 10x7 active window",
              "rows": {"long": len(long_ids), "long_hard": int(hard_long.sum()),
                       "unseen": len(unseen_ids), "unseen_hard": int(hard_unseen.sum())},
              "thresholds_reused_from_bundle_run": thresholds,
              "representations": {}}
    for ix, (name, d, seg, direct_model) in enumerate(eligible, 1):
        print(f"[{ix}/{len(eligible)}] {name}", flush=True)
        long = _eval_config(d, seg, "long", long_ids, hard_long, long_target,
                            rows, pooled, row_lookup, episodes, segsets_list,
                            episodes, segsets_by_name, vq_references, ae_references,
                            direct_model=direct_model)
        unseen = _eval_config(d, seg, "unseen", unseen_ids, hard_unseen, unseen_target,
                              rows, pooled, row_lookup, episodes, segsets_list,
                              un_episodes, unseen_segsets, vq_references, ae_references,
                              unseen=un, direct_model=direct_model)
        result["representations"][name] = {"segmentation": seg,
                                          "family": long["family"],
                                          "code_dim": long["code_dim"],
                                          "stored_values_per_step_mean": long["stored_values_per_step_mean"],
                                          "stored_values_per_step_median": long["stored_values_per_step_median"],
                                          "long": {k: v for k, v in long.items() if k != "predictions"},
                                          "unseen": {k: v for k, v in unseen.items() if k != "predictions"}}
        if ix % 10 == 0:
            print(f"completed {ix}/{len(eligible)} learned bundles", flush=True)

    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
