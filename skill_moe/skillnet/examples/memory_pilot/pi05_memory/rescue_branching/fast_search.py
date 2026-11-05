"""Exact VLM MaxSim token search with adjustable, efficient row blocks."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def normalize_half_keys_blockwise(keys, torch, block_rows=16):
    """Match the existing fp16 normalization without a whole-bank fp32 copy."""
    for start in range(0, int(keys.shape[0]), block_rows):
        end = min(start + block_rows, int(keys.shape[0]))
        norm = torch.linalg.vector_norm(keys[start:end].float(), dim=-1,
            keepdim=True).clamp_min(1e-8).half()
        keys[start:end] /= norm
    return keys


def maxsim_scores_blockwise(keys, key_groups, selected_by_group, ranges, torch, *, block=256):
    """Legacy group-wise MaxSim aggregation over candidate row blocks."""
    scores = torch.empty(len(keys), device=keys.device, dtype=torch.float32)
    for start in range(0, len(keys), int(block)):
        end = min(start + int(block), len(keys))
        terms = []
        for group, (left, right) in enumerate(ranges):
            sim = torch.matmul(selected_by_group[group][None],
                keys[start:end, left:right].transpose(1, 2)).float()
            sim.masked_fill_(key_groups[start:end, None, left:right] != group, -torch.inf)
            terms.append(sim.amax(-1).mean(-1))
        scores[start:end] = torch.stack(terms).mean(0)
    return scores


class FastTokenIndex:
    """TokenIndex-compatible search. `block=32` is the old reference path."""
    def __init__(self, output: Path, *, block=256, norm_block=16, device='cuda'):
        import torch
        self.torch, self.device, self.block = torch, device, int(block)
        self.ids = np.load(output/'memory_row_ids.npy', allow_pickle=False)
        tokens = np.load(output/'tokens.npy', mmap_mode='r', allow_pickle=False)
        group_bank = np.load(output/'groups.npy', mmap_mode='r', allow_pickle=False)
        self.keys = torch.empty((len(self.ids), 712, 2048), device=device, dtype=torch.float16)
        for start in range(0, len(self.ids), norm_block):
            end = min(start + norm_block, len(self.ids))
            rows = torch.as_tensor(np.asarray(tokens[self.ids[start:end]]),
                                   device=device, dtype=torch.float16)
            self.keys[start:end] = normalize_half_keys_blockwise(rows, torch, norm_block)
            del rows
        selected_groups = np.asarray(group_bank[self.ids])
        self.groups = torch.as_tensor(selected_groups, device=device)
        text_valid = np.flatnonzero(np.any(selected_groups[:, 512:] == 2, axis=0))
        self.ranges = [(0, 256), (256, 512), (512, 512 + int(text_valid[-1]) + 1)]
        self.init_seconds = None

    def scores(self, query, query_groups, *, block=None):
        torch = self.torch
        block = self.block if block is None else int(block)
        with torch.inference_mode():
            q = torch.as_tensor(np.asarray(query).copy(), device=self.device).float()
            q = (q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-8)).half()
            query_groups = np.asarray(query_groups)
            selected_by_group = [q[query_groups == group] for group in range(3)]
            if any(not len(selected) for selected in selected_by_group):
                raise ValueError('query is missing a required modality')
            return maxsim_scores_blockwise(self.keys, self.groups, selected_by_group,
                                            self.ranges, torch, block=block)

    def search_topk(self, query, query_groups, k=1, *, block=None):
        scores = self.scores(query, query_groups, block=block)
        # Stable sort matches the legacy argmax's first-row tie break.
        order = self.torch.argsort(scores, descending=True, stable=True)[:min(int(k), len(self.ids))]
        local = order.cpu().numpy()
        values = scores[order].cpu().numpy()
        return [(int(self.ids[i]), float(s)) for i, s in zip(local, values, strict=True)]

    def search(self, query, query_groups):
        return self.search_topk(query, query_groups, 1)[0]

    def legacy_search_topk(self, query, query_groups, k=8, *, block=32):
        """Literal adaptive/BoundedTokenIndex loop, including per-block q slicing."""
        torch=self.torch
        with torch.inference_mode():
            q=torch.as_tensor(np.asarray(query).copy(),device=self.device).float()
            q=(q/torch.linalg.vector_norm(q,dim=-1,keepdim=True).clamp_min(1e-8)).half()
            qgroups=np.asarray(query_groups)
            scores=torch.empty(len(self.ids),device=self.device,dtype=torch.float32)
            for start in range(0,len(self.ids),int(block)):
                end=min(start+int(block),len(self.ids));terms=[]
                for group,(left,right) in enumerate(self.ranges):
                    selected=q[qgroups==group]
                    if not len(selected):raise ValueError('query is missing a required modality')
                    sim=torch.matmul(selected[None],self.keys[start:end,left:right].transpose(1,2)).float()
                    sim.masked_fill_(self.groups[start:end,None,left:right]!=group,-torch.inf)
                    terms.append(sim.amax(-1).mean(-1))
                scores[start:end]=torch.stack(terms).mean(0)
            order=torch.argsort(scores,descending=True,stable=True)[:min(int(k),len(self.ids))]
            local=order.cpu().numpy();values=scores[order].cpu().numpy()
            return [(int(self.ids[i]),float(s)) for i,s in zip(local,values,strict=True)]


def benchmark_one_query(index: FastTokenIndex, query, groups, *, repeats=8):
    """Compare old block width 32 with optimized width 256 on identical keys."""
    reference = index.search_topk(query, groups, 8, block=32)
    fast = index.search_topk(query, groups, 8, block=256)
    legacy = index.legacy_search_topk(query,groups,8,block=32)
    if [x[0] for x in reference] != [x[0] for x in fast]:
        raise AssertionError(f'Top-8 changed: {reference[:3]} != {fast[:3]}')
    if [x[0] for x in legacy] != [x[0] for x in fast]:
        raise AssertionError(f'legacy BoundedTokenIndex changed Top-8: {legacy[:3]} != {fast[:3]}')
    score_delta = max(abs(a[1] - b[1]) for a, b in zip(reference, fast, strict=True))
    legacy_score_delta=max(abs(a[1]-b[1]) for a,b in zip(legacy,fast,strict=True))
    if score_delta > 2e-6:
        raise AssertionError(f'scores differ by {score_delta}')
    timings = {}
    for block in (32, 256):
        samples = []
        for _ in range(repeats + 2):
            started = time.perf_counter()
            index.search_topk(query, groups, 8, block=block)
            samples.append(time.perf_counter() - started)
        timings[str(block)] = {'median_ms': 1000*float(np.median(samples[2:])),
                               'mean_ms': 1000*float(np.mean(samples[2:])),
                               'samples_ms': [1000*x for x in samples[2:]]}
    legacy_samples=[]
    for _ in range(repeats+2):
        started=time.perf_counter();index.legacy_search_topk(query,groups,8,block=32)
        legacy_samples.append(time.perf_counter()-started)
    timings['legacy_block32']={'median_ms':1000*float(np.median(legacy_samples[2:])),
      'mean_ms':1000*float(np.mean(legacy_samples[2:])),
      'samples_ms':[1000*x for x in legacy_samples[2:]]}
    timings['fast_block256']=timings['256']
    return {'same_top8_ids': True, 'max_abs_top8_score_delta': score_delta,
            'legacy_same_top8_ids':True,'max_abs_legacy_score_delta':legacy_score_delta,
            'top1_row': reference[0][0], 'top1_score': reference[0][1],
            'timings': timings}
