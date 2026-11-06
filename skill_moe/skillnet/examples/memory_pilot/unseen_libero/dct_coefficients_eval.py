#!/usr/bin/env python3
"""Repeat the stored-unit comparison with other numbers of DCT values per axis."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory'), str(HERE.parent / 'pi05_memory' / 'shape_skills')]
import dct_skill_draft_eval as ev  # noqa: E402
import optimal_units_eval as opt  # noqa: E402
import stored_units_eval as stored  # noqa: E402
import variable_shape_memory as vs  # noqa: E402
from draft_reconstruction import ROOT  # noqa: E402

COEFFICIENTS = (2, 3, 4, 5)
SEGMENTS = ('fixed10', 'e0.005')


def run_one(config):
    k, segment, level, stretch = config
    vs.set_coefficients(k)
    entries, _ = opt.run_one((segment, level, stretch))
    return [{'k': k, **entry} for entry in entries]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'unseen-libero' / 'draft-reconstruction-01' / 'dct_coefficients.json')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    stored.load_data()
    opt.FRAMES_PER_SELECTION = (10, 20)
    configs = [(k, s, l, t) for k in COEFFICIENTS for s in SEGMENTS for l in opt.LEVELS for t in opt.STRETCHES]
    with ProcessPoolExecutor(args.workers) as pool:
        results = [e for part in pool.map(run_one, configs) for e in part]
    args.output.write_text(json.dumps({'rows': len(ev.Shared.target), 'hard_rows': int(ev.Shared.hard.sum()), 'results': results}, ensure_ascii=False, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
