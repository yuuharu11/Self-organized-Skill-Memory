"""Measure MaxSim search and assert rank/score equivalence to block=32."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
import argparse

import numpy as np

from fast_search import FastTokenIndex, benchmark_one_query

RUN = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
OUT = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/rescue-branching-01')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', default='corrected')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    torch.cuda.set_per_process_memory_fraction(.30, 0)
    started = time.perf_counter()
    index = FastTokenIndex(RUN, block=256, norm_block=16)
    torch.cuda.synchronize()
    init_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    base_allocated = torch.cuda.memory_allocated()
    rows = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False)
            for p in (RUN/'rows').glob('*.npy')}
    query_row = int(np.flatnonzero(rows['validation'])[0])
    tokens = np.load(RUN/'tokens.npy', mmap_mode='r', allow_pickle=False)
    groups = np.load(RUN/'groups.npy', mmap_mode='r', allow_pickle=False)
    result = benchmark_one_query(index, np.asarray(tokens[query_row]),
                                 np.asarray(groups[query_row]), repeats=8)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    free, total = torch.cuda.mem_get_info()
    result.update({'query_row': query_row, 'index_init_seconds': init_seconds,
        'GPU_allocated_index_GiB': float(index.keys.numel()*2/(1024**3)),
        'GPU_max_allocated_GiB': float(peak/(1024**3)),
        'GPU_search_peak_extra_GiB': float((peak-base_allocated)/(1024**3)),
        'total_GPU_mem_used_MiB': int(total-free)//(1024**2),
        'block_old':32,'block_fast':256})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f'search_benchmark_{args.suffix}.json').write_text(json.dumps(result, indent=2)+'\n')
    report = ['# 検索ブロック幅の高速化', '',
        'run-01 の固定validation行1件を同じ正規化済みメモリbankで検索。block=32（旧）と256（新）を比較した。',
        'Top-8行IDが完全一致し、最大score差は許容2e-6以下であることをassertした。', '',
        '| block | median ms/query | mean ms/query |', '|---:|---:|---:|']
    for block in ('32','256'):
        t = result['timings'][block]
        report.append(f"| {block} | {t['median_ms']:.2f} | {t['mean_ms']:.2f} |")
    report += ['', f"Top-1 row={result['top1_row']}, score={result['top1_score']:.8f}; max Top-8 score delta={result['max_abs_top8_score_delta']:.2g}.",
        f"Index tensor={result['GPU_allocated_index_GiB']:.2f} GiB; measured peak allocated={result['GPU_max_allocated_GiB']:.2f} GiB; search peak extra after index initialization={result['GPU_search_peak_extra_GiB']:.2f} GiB. See search_benchmark_{args.suffix}.json for full samples."]
    (OUT/f'search_benchmark_{args.suffix}.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
