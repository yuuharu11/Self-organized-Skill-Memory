#!/usr/bin/env python3
"""Check drafts on unseen demos before any denoising.

Units are selected with the correct actions and followed to their end. Direction and
size come from the correct actions, or from the normal pi0.5 output when it is cached.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dct_skill_draft import DraftPlanner, SkillMemory, oracle_selector  # noqa: E402
from draft_reconstruction import ROOT, RUN, UNSEEN, UNSEEN_ROWS, summarize  # noqa: E402
from variable_length_error import stitch  # noqa: E402

HORIZON = 10
CONFIGS = [('e0.005', .04), ('e0.005', .08), ('e0.005', .16), ('fixed10', .04), ('fixed10', .08), ('fixed10', .16)]
# (direction and size set per window or per unit, gripper must match to fit)
VARIANTS = [('window', False), ('unit', False), ('window', True), ('unit', True)]


class Shared:
    """Data the worker processes inherit."""


def roll(memory, trace, tau, references, *, pose_per='window', match_gripper=False):
    """Draft every horizon of one demo. `references` is [windows,10,6] or None for correct."""
    unit_reference = (lambda first, length: trace[first:first + length, :6]) if pose_per == 'unit' else None
    planner = DraftPlanner(memory, oracle_selector(memory, trace, tau, match_gripper=match_gripper),
                           unit_reference=unit_reference)
    drafts, starts = [], []
    for w in range(len(trace) // HORIZON):
        reference = trace[w * HORIZON:(w + 1) * HORIZON] if references is None else references[w]
        drafts.append(planner.draft(reference))
        for unit, first in planner.plan:
            starts.append((unit, first))
        planner.advance(HORIZON)
    used = sorted(set(starts), key=lambda s: s[1])
    return np.stack(drafts), used, planner.selections


def evaluate(config):
    segment, tau = config
    memory = SkillMemory.build(list(Shared.memory.values()), segment=segment, tau=tau)
    out = []
    runs = [('correct', pose, grip) for pose, grip in VARIANTS]
    if Shared.normal is not None:
        runs += [('pi05', 'window', grip) for grip in (False, True)]
    for source, pose_per, match_gripper in runs:
        pred = np.empty_like(Shared.target, dtype=np.float64)
        selections, frames, jumps, base_jumps, lengths = 0, 0, [], [], []
        for key, trace in Shared.traces.items():
            rows = Shared.rows_of[key]
            references = None if source == 'correct' else Shared.normal[rows][:, :, :6]
            drafts, used, chosen = roll(memory, trace, tau, references, pose_per=pose_per, match_gripper=match_gripper)
            pred[rows] = drafts
            selections += chosen
            frames += len(trace)
            lengths += [int(memory.lengths[u]) for u, _ in used]
            joined = drafts.reshape(-1, 7)
            for _, first in used[1:]:
                if 0 < first < len(joined):
                    jumps.append(np.mean((joined[first, :6] - joined[first - 1, :6]) ** 2))
                    base_jumps.append(np.mean((trace[first, :6] - trace[first - 1, :6]) ** 2))
        out.append({'segment': segment, 'tau': tau, 'source': source, 'pose_per': pose_per,
                    'match_gripper': match_gripper, 'units': len(memory.values),
                    'selections': selections, 'frames_per_selection': frames / selections,
                    'length_median': float(np.median(lengths)), 'jump': float(np.mean(jumps)),
                    'jump_in_demo': float(np.mean(base_jumps)), **summarize(pred, Shared.target, Shared.hard)})
        print(json.dumps({k: out[-1][k] for k in ('segment', 'tau', 'source', 'pose_per', 'match_gripper', 'frames_per_selection', 'all', 'hard')}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'unseen-libero' / 'draft-reconstruction-01' / 'skill_draft.json')
    args = ap.parse_args()
    rows = {k: np.load(RUN / 'rows' / f'{k}.npy', mmap_mode='r') for k in ('actions', 'task', 'episode', 'frame', 'memory')}
    memory = np.flatnonzero(np.asarray(rows['memory'], bool))
    key = np.stack([np.asarray(rows['task'])[memory], np.asarray(rows['episode'])[memory]], 1)
    Shared.memory, _ = stitch(np.asarray(rows['actions'][memory]), key, np.asarray(rows['frame'])[memory])
    un = {k: np.load(UNSEEN_ROWS / f'{k}.npy', mmap_mode='r') for k in ('actions', 'task', 'episode', 'frame', 'is_dev')}
    ev = np.flatnonzero(~np.asarray(un['is_dev']))
    raw = np.load(UNSEEN / 'drafts' / 'raw_none_eval.npz', allow_pickle=False)
    if not np.array_equal(raw['query_ids'], ev + len(rows['task'])):
        raise AssertionError('unseen eval rows differ from the draft cache')
    threshold = json.loads((UNSEEN / 'hard_threshold.json').read_text())['threshold']
    Shared.hard = raw['pre_mse'] >= threshold
    ukey = np.stack([np.asarray(un['task']), np.asarray(un['episode'])], 1)[ev]
    frame = np.asarray(un['frame'])[ev]
    Shared.traces, _ = stitch(np.asarray(un['actions'])[ev], ukey, frame)
    Shared.rows_of = {k: np.flatnonzero((ukey == k).all(1))[np.argsort(frame[(ukey == k).all(1)])] for k in Shared.traces}
    Shared.target = np.asarray(un['actions'])[ev][:, :, :7].astype(np.float64)
    Shared.normal = None
    normal_path = UNSEEN / 'official_normal_eval.npz'
    extra = {}
    if normal_path.exists():
        z = np.load(normal_path, allow_pickle=False)
        if len(z['row_ids']) != len(ev) or not np.allclose(z['target'], Shared.target, atol=1e-5):
            raise AssertionError('the normal output rows differ from the evaluation rows')
        Shared.normal = np.asarray(z['prediction'], np.float64)
        extra['normal_pi05'] = summarize(Shared.normal[:, :, :7], Shared.target, Shared.hard)
    with ProcessPoolExecutor(len(CONFIGS)) as pool:
        results = [e for part in pool.map(evaluate, CONFIGS) for e in part]
    args.output.write_text(json.dumps({'rows': len(ev), 'hard_rows': int(Shared.hard.sum()), **extra, 'results': results}, ensure_ascii=False, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
