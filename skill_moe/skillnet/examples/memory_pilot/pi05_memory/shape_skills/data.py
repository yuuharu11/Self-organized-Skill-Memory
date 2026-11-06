"""Read-only frame/row data and label-free variable-length segmentation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


LONG_TASKS = tuple(range(10))
HELDOUT_TASKS = (17, 19, 28, 29, 35, 38)


def training_row_mask(rows):
    """Rows allowed to fit transforms/codec: train split, non-Long, non-heldout."""
    task=np.asarray(rows["task"])
    return ((task>=10)&~np.isin(task,HELDOUT_TASKS)&
            ~np.asarray(rows["validation"],bool)&~np.asarray(rows["memory"],bool))


def qstats(checkpoint):
    raw = json.loads((Path(checkpoint) / "assets/physical-intelligence/libero/norm_stats.json").read_text())
    x = raw["norm_stats"]["actions"]
    return np.asarray(x["q01"], np.float32), np.asarray(x["q99"], np.float32)


def normalize_physical(a, q01, q99):
    # Match the run-01 affine q01/q99 transform; values outside the quantiles
    # remain outside [-1,1] (the source action cache is not clipped).
    return (np.asarray(a, np.float32) - q01) / (q99 - q01 + 1e-6) * 2 - 1


def unnormalize(a, q01, q99):
    return (np.asarray(a, np.float32) + 1) * .5 * (q99 - q01 + 1e-6) + q01


def load_source(run_dir, checkpoint):
    """Load only actions/state/indices from the source parquet, never images."""
    import pyarrow.parquet as pq

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "data_manifest.json").read_text())
    source = Path(manifest["source"])
    q01, q99 = qstats(checkpoint)
    rows = {p.stem: np.load(p, mmap_mode="r", allow_pickle=False)
            for p in (run_dir / "rows").glob("*.npy")}
    episode_meta = {}
    pair_to_ix = {}
    for i, (task, episode) in enumerate(zip(rows["task"], rows["episode"], strict=True)):
        pair_to_ix.setdefault((int(task), int(episode)), []).append(i)
    for key, indices in pair_to_ix.items():
        ix = np.asarray(indices, np.int64)
        mem = np.asarray(rows["memory"][ix], bool)
        val = np.asarray(rows["validation"][ix], bool)
        if np.any(mem != mem[0]) or np.any(val != val[0]):
            raise AssertionError(f"episode split flags are inconsistent: {key}")
        episode_meta[key] = {"memory": bool(mem[0]), "validation": bool(val[0]), "present": True}
    episodes = {}
    paths = sorted(source.glob("data/chunk-*/episode_*.parquet"))
    for path in paths:
        table = pq.read_table(path, columns=["actions", "state", "frame_index", "task_index", "episode_index"])
        def fixed_list_matrix(column):
            arr=column.combine_chunks()
            if hasattr(arr.type,"list_size") and arr.type.list_size is not None:
                return np.asarray(arr.values.to_numpy(zero_copy_only=False),np.float32).reshape(len(arr),arr.type.list_size)
            return np.asarray(arr.to_numpy(zero_copy_only=False),np.float32)
        a = fixed_list_matrix(table["actions"])
        s = fixed_list_matrix(table["state"])
        frame = np.asarray(table["frame_index"].to_numpy(), np.int64)
        task = int(np.asarray(table["task_index"].to_numpy())[0])
        episode = int(np.asarray(table["episode_index"].to_numpy())[0])
        order = np.argsort(frame, kind="stable")
        frame, a, s = frame[order], a[order], s[order]
        if not np.array_equal(frame, np.arange(len(frame), dtype=np.int64)):
            raise AssertionError(f"non-contiguous frame indexes: {path}")
        # LeRobot action values are physical OSC deltas; keep them physical until
        # the last step of a representation transform.
        key = (task, episode)
        episodes[key] = {"task": task, "episode": episode, "frame": frame,
                          "actions_phys": a, "actions_norm": normalize_physical(a, q01, q99),
                          "state_phys": s, "meta": episode_meta.get(key, {"memory": False, "validation": False, "present": False})}
    if not episodes:
        raise FileNotFoundError(f"no source parquet under {source}")
    # Source-row parity: every stored 10-step action window must map to the same
    # physical sample sequence after q01/q99 normalization.
    row_actions = np.load(run_dir / "rows/actions.npy", mmap_mode="r", allow_pickle=False)
    checked = 0
    for rid, (task, episode, start) in enumerate(zip(rows["task"], rows["episode"], rows["frame"], strict=True)):
        seq = episodes.get((int(task), int(episode)))
        if seq is None or int(start) + 10 > len(seq["frame"]):
            continue
        if checked < 64:
            expected = seq["actions_norm"][int(start):int(start)+10]
            np.testing.assert_allclose(expected, row_actions[rid, :, :7], atol=4e-5, rtol=0)
            checked += 1
    if checked < 16:
        raise AssertionError(f"too few source/run action parity windows: {checked}")
    return episodes, rows, {"source": str(source), "parquet_episodes": len(episodes),
                            "source_row_parity_checked": checked,
                            "q01": q01.tolist(), "q99": q99.tolist()}


def _merge_short(cuts, n, minimum=10):
    cuts = sorted(set([0, *[int(x) for x in cuts if 0 < int(x) < n], n]))
    out = [0]
    for end in cuts[1:-1]:
        if end - out[-1] >= minimum and n - end >= minimum:
            out.append(end)
    out.append(n)
    # If the episode itself is shorter than minimum, keep it intact.
    return [(a, b) for a, b in zip(out[:-1], out[1:])]


def fixed10(seq, minimum=10):
    n = len(seq["frame"])
    # Exactly the dataset's non-overlapping stride-10 windows; any incomplete
    # terminal remainder is not a complete action chunk and is omitted.
    return [(start,start+10) for start in range(0,n-9,10)]


def _constant_error(a, b):
    x = a[b[0]:b[1], :7]
    mean = x.mean(axis=0, keepdims=True)
    return float(np.mean((x - mean) ** 2))


def threshold_segments(seq, eps, method="greedy", minimum=10, *, return_fallback=False):
    """Piecewise-constant normalized-action reconstruction with MSE <= eps.

    DP minimizes segment count under the error bound. If no legal path exists
    (e.g. even a minimum-length piece exceeds epsilon), it falls back to the
    fixed minimum-window baseline rather than claiming the whole episode is a
    valid low-error segment.
    """
    x = seq["actions_norm"][:, :7]
    n = len(x)
    if n < minimum:
        result = [(0, n)]
        return (result, False) if return_fallback else result
    if method == "greedy":
        cuts = [0]
        start = 0
        while start + minimum < n:
            valid = start + minimum
            for end in range(start + minimum, n + 1):
                if _constant_error(x, (start, end)) <= eps:
                    valid = end
                else:
                    break
            if valid <= start:
                valid = min(start + minimum, n)
            if n - valid < minimum:
                valid = n
            cuts.append(valid)
            start = valid
        result = list(zip(cuts[:-1], cuts[1:]))
        return (result, False) if return_fallback else result
    if method != "dp":
        raise ValueError(method)
    # DP minimizes segment count; ties choose the longest first segment.
    best = [10**9] * (n + 1)
    nxt = [-1] * (n + 1)
    best[n] = 0
    for start in range(n - minimum, -1, -1):
        for end in range(n, start + minimum - 1, -1):
            if end < n and n - end < minimum:
                continue
            if best[end] == 10**9:
                continue
            if _constant_error(x, (start, end)) <= eps and best[end] + 1 < best[start]:
                best[start], nxt[start] = best[end] + 1, end
    if nxt[0] < 0:
        result = fixed10({"frame":np.arange(n)}) if minimum==10 else [
            (a,min(a+minimum,n)) for a in range(0,n-minimum+1,minimum)]
        return (result, True) if return_fallback else result
    result, start = [], 0
    while start < n:
        end = nxt[start]
        if end <= start:
            return [(0, n)]
        result.append((start, end)); start = end
    return (result, False) if return_fallback else result


def grip_segments(seq, minimum=10):
    g = seq["actions_phys"][:, 6]
    sign = g >= 0
    cuts = np.flatnonzero(sign[1:] != sign[:-1]) + 1
    return _merge_short(cuts, len(g), minimum)


def _moving_average(x, width=7):
    kernel = np.ones(width, np.float64) / width
    return np.convolve(np.asarray(x, np.float64), kernel, mode="same")


def grip_velocity_segments(seq, threshold, minimum=10):
    a = seq["actions_phys"]
    speed = _moving_average(np.linalg.norm(a[:, :3], axis=1), 7)
    local = np.flatnonzero((speed[1:-1] <= speed[:-2]) & (speed[1:-1] < speed[2:]) &
                           (speed[1:-1] <= threshold)) + 1
    grip = np.flatnonzero((a[1:, 6] >= 0) != (a[:-1, 6] >= 0)) + 1
    return _merge_short(np.concatenate([local, grip]), len(a), minimum)


def choose_velocity_threshold(episodes, dev_keys):
    speeds = []
    for key in dev_keys:
        speeds.extend(np.linalg.norm(episodes[key]["actions_phys"][:, :3], axis=1).tolist())
    if not speeds:
        raise ValueError("no heldout Dev episodes for velocity threshold selection")
    q = np.quantile(speeds, [.1, .2, .3, .4, .5, .6])
    records = []
    for threshold in q:
        lens = []
        for key in dev_keys:
            lens.extend(b-a for a,b in grip_velocity_segments(episodes[key], float(threshold)))
        median = float(np.median(lens)) if lens else float("inf")
        records.append((abs(np.clip(median, 10, 40)-25), float(threshold), median))
    _, threshold, median = min(records)
    return threshold, {"candidate_thresholds": [[v, m] for _,v,m in records],
                       "selected_threshold": threshold, "dev_median_segment_frames": median}


def type_change_segments(seq, threshold, minimum=10):
    a = seq["actions_phys"][:, :6]
    n = len(a)
    # Compare unit mean motion over adjacent 10-frame neighborhoods. These are
    # unlabeled kinematic descriptors, not language/skill annotations.
    left = np.zeros_like(a); right = np.zeros_like(a)
    for i in range(n):
        left[i] = a[max(0, i-10):i+1].mean(axis=0)
        right[i] = a[i:min(n, i+11)].mean(axis=0)
    ln = np.linalg.norm(left, axis=1); rn = np.linalg.norm(right, axis=1)
    cosine = np.sum(left*right, axis=1) / np.maximum(ln*rn, 1e-9)
    change = 1 - np.clip(cosine, -1, 1)
    peaks = (change[1:-1] >= change[:-2]) & (change[1:-1] > change[2:]) & (change[1:-1] >= threshold)
    return _merge_short(np.flatnonzero(peaks)+1, n, minimum)


def choose_type_threshold(episodes, dev_keys):
    values = []
    for key in dev_keys:
        seg = episodes[key]
        a = seg["actions_phys"][:, :6]
        means = np.stack([a[max(0,i-10):i+1].mean(0) for i in range(len(a))])
        nxt = np.stack([a[i:min(len(a),i+11)].mean(0) for i in range(len(a))])
        cos = np.sum(means*nxt, axis=1)/np.maximum(np.linalg.norm(means,axis=1)*np.linalg.norm(nxt,axis=1),1e-9)
        values.extend((1-np.clip(cos,-1,1)).tolist())
    qs = np.quantile(values, [.7,.8,.9,.95])
    records=[]
    for q in qs:
        lens=[]
        for key in dev_keys:
            lens.extend(b-a for a,b in type_change_segments(episodes[key], float(q)))
        med=float(np.median(lens)) if lens else float("inf")
        records.append((abs(np.clip(med,10,40)-25),float(q),med))
    _,threshold,median=min(records)
    return threshold,{"candidate_thresholds":[[v,m] for _,v,m in records],
                      "selected_threshold":threshold,"dev_median_segment_frames":median}
