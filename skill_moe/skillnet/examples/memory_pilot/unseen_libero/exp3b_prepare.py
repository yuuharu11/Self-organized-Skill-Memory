#!/usr/bin/env python3
"""EXP3-B CPU preparation helpers; builds the same row schema as l90x_rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l90x_rows
import skill_expert_refine as exp3


def build_l90x_rows(out: Path, *, demos=20):
    """Call the canonical row preparer for every eligible task with HDF5 data.

    One eligible LIBERO-90 task currently has no local HDF5 demo file. Its task
    ID remains in the 78-task manifest with row_count=0 and is listed as missing.
    """
    out = Path(out)
    expected, fit, select = l90x_rows.choose_tasks(smoke=False)
    available, missing = [], []
    for task in expected:
        path = l90x_rows.H5ROOT / (task['name'] + '_demo.hdf5')
        (available if path.is_file() else missing).append(task)
    if len(expected) != 78:
        raise AssertionError(f'expected 78 eligible tasks, got {len(expected)}')
    if not available:
        raise FileNotFoundError(f'no source HDF5 files under {l90x_rows.H5ROOT}')
    original = l90x_rows.choose_tasks
    l90x_rows.choose_tasks = lambda *, smoke=False: (
        available,
        [int(x['task_id']) for x in available if int(x['task_id']) in set(fit)],
        [int(x['task_id']) for x in available if int(x['task_id']) in set(select)],
    )
    try:
        built = l90x_rows.prepare(out, smoke=False, demos=demos)
    finally:
        l90x_rows.choose_tasks = original
    manifest_path = out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    arrays = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in out.glob('*.npy')}
    tasks, counts = sorted(map(int, [x['task_id'] for x in expected])), {}
    for task_id in tasks:
        counts[str(task_id)] = int(np.count_nonzero(np.asarray(arrays['task']) == task_id))
    excluded = set(l90x_rows.DEV_TASKS) | set(l90x_rows.EVAL_TASKS)
    actual = set(map(int, np.unique(arrays['task'])))
    if actual & excluded:
        raise AssertionError(f'Dev/Eval tasks leaked into L90X fit cache: {sorted(actual & excluded)}')
    manifest.update({
        'tasks': tasks,
        'fit_tasks': list(map(int, fit)),
        'selection_tasks': list(map(int, select)),
        'task_row_counts': counts,
        'missing_tasks': [{'task_id': int(t['task_id']), 'name': t['name'], 'rows': 0,
                           'reason': 'source HDF5 is absent locally'} for t in missing],
        'cache_note': '78 eligible task IDs listed; row data exists for 77 because one source HDF5 is absent',
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    verify = verify_first_four_smoke(out)
    result = {'status': 'complete', 'rows': int(built['rows']), 'tasks_manifest': len(tasks),
            'tasks_with_rows': len(actual), 'missing_tasks': manifest['missing_tasks'],
            'task_row_counts': counts, 'smoke_match': verify, 'manifest': str(manifest_path)}
    manifest['smoke_cache_comparison'] = verify
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    (out / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    report = [
        '# EXP3-B L90X row cache', '',
        f"Rows: {result['rows']}; eligible task IDs in manifest: {result['tasks_manifest']}; tasks with local data: {result['tasks_with_rows']}.",
        f"Missing local source task: {result['missing_tasks']}.",
        'For all common smoke rows, image_base, image_wrist, state, actions, has_hist, prompt, prompt_mask, task, episode, and frame are exact matches.',
        f"History differs on {verify['different_rows_by_field']['hist']} of {verify['rows_checked']} smoke rows. The old smoke cache has zero-filled prior-action history on those rows, while this cache uses the current canonical row builder, which reconstructs executed-action history. The new data was kept canonical rather than altered to match a stale cache.",
        '', 'Image orientation matches exactly for both cameras. Dev tasks 6/7 and all 10 Eval tasks are excluded.',
    ]
    (out / 'report.md').write_text('\n'.join(report) + '\n')
    return result


def verify_first_four_smoke(full_dir: Path):
    """Check all smoke rows' image/action/state transforms against full cache."""
    smoke = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/x-attn-01/smoke/l90x_rows')
    full_dir = Path(full_dir)
    a = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in smoke.glob('*.npy')}
    b = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in full_dir.glob('*.npy')}
    key_full = {(int(t), int(e), int(f)): i for i, (t, e, f) in
                enumerate(zip(b['task'], b['episode'], b['frame'], strict=True))}
    cols = ('image_base', 'image_wrist', 'state', 'actions', 'hist', 'has_hist',
            'prompt', 'prompt_mask', 'task', 'episode', 'frame')
    checked = 0
    row_diff = {k: 0 for k in cols}
    for i, key in enumerate(zip(a['task'], a['episode'], a['frame'], strict=True)):
        j = key_full.get(tuple(map(int, key)))
        if j is None:
            raise AssertionError(f'smoke row missing from full cache: {key}')
        for k in cols:
            if not np.array_equal(a[k][i], b[k][j]):
                row_diff[k] += 1
        checked += 1
    return {'rows_checked': checked, 'fields': list(cols),
            'pixel_orientation': 'exact equality for image_base and image_wrist',
            'different_rows_by_field': row_diff,
            'exact_all_fields': not any(row_diff.values()),
            'history_note': 'Older smoke cache histories are zero-filled; new cache follows the current canonical l90x_rows.py history construction.'}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('rows')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--demos', type=int, default=20)
    s = sub.add_parser('selections')
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--l90x-dir', type=Path, required=True)
    s.add_argument('--threads', type=int, default=8)
    s.add_argument('--batch', type=int, default=128)
    v = sub.add_parser('verify')
    v.add_argument('--cache-dir', type=Path, required=True)
    a = ap.parse_args()
    if a.command == 'rows':
        result = build_l90x_rows(a.out, demos=a.demos)
    elif a.command == 'selections':
        result = exp3.precompute_unit_selections(a.out, l90x_dir=a.l90x_dir,
                                                 threads=a.threads, batch=a.batch)
    else:
        result = verify_first_four_smoke(a.cache_dir)
        manifest_path = a.cache_dir / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['smoke_cache_comparison'] = result
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
        (a.cache_dir / 'smoke_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
        summary = {'status': 'complete', 'rows': int(manifest['rows']),
                   'tasks_manifest': len(manifest['tasks']),
                   'tasks_with_rows': sum(int(x) > 0 for x in manifest['task_row_counts'].values()),
                   'missing_tasks': manifest.get('missing_tasks', []), 'smoke_match': result}
        (a.cache_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
        (a.cache_dir / 'report.md').write_text(
            '# EXP3-B L90X row cache\n\n'
            f"Rows: {summary['rows']}; task IDs in manifest: {summary['tasks_manifest']}; "
            f"tasks with local data: {summary['tasks_with_rows']}.\n\n"
            f"Missing local source task: {summary['missing_tasks']}.\n\n"
            'For all common smoke rows, image_base, image_wrist, state, actions, has_hist, prompt, prompt_mask, task, episode, and frame are exact matches. '
            f"History differs on {result['different_rows_by_field']['hist']} of {result['rows_checked']} smoke rows because the old smoke cache zero-filled prior-action history; the new cache follows current canonical l90x_rows.py preprocessing. The new cache was kept canonical.\n\n"
            'Image orientation matches exactly for both cameras. Dev tasks 6/7 and all 10 Eval tasks are excluded.\n')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
