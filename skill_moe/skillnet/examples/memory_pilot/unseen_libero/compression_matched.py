#!/usr/bin/env python3
"""Value-count-matched compression diagnostics for unseen LIBERO.

The script is CPU-only.  All representation fitting (PCA and VQ) uses LIBERO-40
rows with validation=False.  LIBERO-90 targets are only used to select the closest
already-built template and to score reconstructions.  `--smoke` limits the scored
windows and the diagnostic fitting sample; it still runs the exact variable DCT3
reproduction check before any other analysis.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct
from sklearn.cluster import MiniBatchKMeans

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory')]
import dct_skill_draft_eval as ev  # noqa: E402
import motion_core_eval as motion  # noqa: E402
import stored_units_eval as stored  # noqa: E402
from draft_reconstruction import (ROOT, RUN, UNSEEN_ROWS, dct_roundtrip,
                                  keyframe_roundtrip, pca_basis, pca_roundtrip,
                                  summarize)  # noqa: E402
from variable_length_error import stitch  # noqa: E402

OUT_DEFAULT = ROOT / 'unseen-libero' / 'compress-matched-01'
VALUE_COUNTS = (14, 21, 28)
TAUS = (None, .04, .08, .16)
SEED = 20261114
BASELINE = {'units': 4516, 'all': .034447897620114326, 'hard': .06845424480130637}


def dct_encode(x: np.ndarray, k: int) -> np.ndarray:
    """Orthonormal DCT coefficients, independently per action dimension."""
    a = np.asarray(x, np.float64)
    return dct(a, type=2, norm='ortho', axis=-2)[..., :k, :]


def dct_decode(coeff: np.ndarray, length: int) -> np.ndarray:
    c = np.zeros((*coeff.shape[:-2], length, coeff.shape[-1]), np.float64)
    n = min(length, coeff.shape[-2])
    c[..., :n, :] = coeff[..., :n, :]
    return idct(c, type=2, norm='ortho', axis=-2)


def keyframe_encode(x: np.ndarray, points: int) -> np.ndarray:
    """Sample evenly spaced points (including endpoints) per action axis."""
    ids = np.rint(np.linspace(0, x.shape[-2] - 1, points)).astype(int)
    return np.asarray(x)[..., ids, :]


def keyframe_decode(values: np.ndarray, length: int = 10) -> np.ndarray:
    xp = np.linspace(0., 1., values.shape[-2])
    x = np.linspace(0., 1., length)
    flat = np.asarray(values).reshape(-1, values.shape[-2], values.shape[-1])
    out = np.empty((len(flat), length, flat.shape[-1]), np.float64)
    for i, row in enumerate(flat):
        for j in range(row.shape[-1]):
            out[i, :, j] = np.interp(x, xp, row[:, j])
    return out.reshape((*values.shape[:-2], length, values.shape[-1]))


def pca_fit(x: np.ndarray, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """Stable covariance eigensystem for flattened 10x7 windows."""
    a = np.asarray(x, np.float64).reshape(len(x), -1)
    mean = a.mean(axis=0)
    centered = a - mean
    # The feature dimension is only 70; this avoids materializing a large SVD U.
    cov = centered.T @ centered
    _, _, vt = np.linalg.svd(cov, full_matrices=False)
    return mean, vt[:dims]


def pca_project(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    flat = np.asarray(x, np.float64).reshape(len(x), -1)
    return (flat - mean) @ basis.T


def pca_restore(z: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if mean.size % 7:
        raise ValueError(f'PCA feature width must be divisible by active7, got {mean.size}')
    return (z @ basis + mean).reshape(-1, mean.size // 7, 7)


def greedy_cover(decoded: np.ndarray, tau: float | None) -> np.ndarray:
    if tau is None:
        return np.arange(len(decoded))
    flat = np.asarray(decoded, np.float32).reshape(len(decoded), -1)
    # Match the existing greedy cover implementation and its inclusive threshold.
    within = np.empty((len(flat), len(flat)), bool)
    for i in range(len(flat)):
        within[i] = np.mean((flat - flat[i]) ** 2, axis=1) <= tau
    import dct_tolerance_merge as merge
    return np.asarray(merge.greedy_cover(within)[0], dtype=np.int64)


def nearest_templates(target: np.ndarray, templates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return closest template indices and per-query active7 MSE."""
    t = np.asarray(target, np.float32)
    c = np.asarray(templates, np.float32)
    chosen = np.empty(len(t), np.int64)
    errors = np.empty(len(t), np.float64)
    for i in range(len(t)):
        e = np.mean((c - t[i]) ** 2, axis=(1, 2))
        chosen[i] = int(np.argmin(e))
        errors[i] = float(e[chosen[i]])
    return chosen, errors


def variable_templates(memory: dict, min_length: int = motion.MIN_LENGTH):
    """Anchor cuts (speed minima + gripper changes), encoded and decoded in place."""
    acts, lengths, keys = [], [], []
    for key, trace in memory.items():
        for start, end in motion.anchor_spans(trace, min_length=min_length):
            a = np.asarray(trace[start:end, :7], np.float64)
            co = motion.encode(trace, None, start, end, 'world', 1)
            dec = motion.decode(co[None], len(a), 1)[0]
            acts.append(a)
            lengths.append(end - start)
            keys.append(tuple(map(int, key)))
            # exact reversible check for the full DCT length is not expected; the
            # returned coefficients are used to build compressed variants below.
            del dec
    return acts, np.asarray(lengths), np.asarray(keys, dtype=np.int64)


def compressed_variable(actions: list[np.ndarray], kind: str, amount: int,
                        mean: np.ndarray | None = None, basis: np.ndarray | None = None) -> list[np.ndarray]:
    """Encode/decode variable units; each returned item retains its original duration."""
    out = []
    for a in actions:
        if kind == 'dct':
            coded = dct_encode(a, min(amount, len(a)))
            rec = dct_decode(coded, len(a))
        elif kind == 'pca':
            assert mean is not None and basis is not None
            rec = phase_pca_restore(a, mean, basis[:amount], len(a))
        elif kind == 'raw32':
            # Store 32 phase samples, then restore to this unit's own duration.
            rec = phase_resample(phase_resample(a, 32), len(a))
        else:
            raise ValueError(kind)
        out.append(np.asarray(rec, np.float32))
    return out


def cover_variable(items: list[np.ndarray], tau: float | None) -> np.ndarray:
    if tau is None:
        return np.arange(len(items), dtype=np.int64)
    phase = np.stack([phase_resample(x, 32).reshape(-1) for x in items]).astype(np.float32)
    norms = np.sum(phase * phase, axis=1)
    dist = (norms[:, None] + norms[None, :] - 2 * (phase @ phase.T)) / phase.shape[1]
    dist = np.maximum(dist, 0.)
    import dct_tolerance_merge as merge
    return np.asarray(merge.greedy_cover(dist <= tau)[0], np.int64)


def equal_count_samples(groups: dict[str, list[np.ndarray]], seed: int = SEED) -> tuple[dict[str, list[np.ndarray]], int]:
    """Deterministically subsample each segmentation to the same unit count."""
    if not groups:
        return {}, 0
    count = min(map(len, groups.values()))
    if count == 0:
        return {name: [] for name in groups}, 0
    rng = np.random.default_rng(seed)
    sampled = {}
    for name, items in groups.items():
        ids = np.sort(rng.choice(len(items), count, replace=False))
        sampled[name] = [items[int(i)] for i in ids]
    return sampled, count


def score_variable_memory(items: list[np.ndarray], lengths: np.ndarray, units: np.ndarray,
                          eval_traces: dict, rows_of: dict) -> tuple[np.ndarray, dict]:
    """Variable playback, with a DP budget of one new selection per ten frames."""
    import optimal_units_eval as opt
    groups: dict[int, list[np.ndarray]] = {}
    for u in units:
        a = items[int(u)]
        groups.setdefault(int(lengths[int(u)]), []).append(a)
    drafts = {n: np.stack(v).astype(np.float64) for n, v in groups.items()}
    keys = sorted(drafts)
    pred = np.empty_like(ev.Shared.target, dtype=np.float64)
    selections = frames = 0
    for key, trace in eval_traces.items():
        table = opt.best_cost_table(trace, drafts)
        budget = max(1, math.ceil(len(trace) / 10))
        _, seq = opt.best_sequence(table, keys, len(trace), budget)
        joined = np.empty_like(trace, dtype=np.float64)
        for start, length in seq:
            piece = table[start, length][1]
            joined[start:start + len(piece)] = piece
        pred[rows_of[key]] = joined.reshape(-1, 10, 7)
        selections += len(seq)
        frames += len(trace)
    return pred, {'selections': selections, 'frames_per_selection': frames / max(1, selections)}


def balanced_task_subsets(task_ids: np.ndarray) -> dict:
    """Three deterministic 20/10-task subsets, stratified over four task-ID bands."""
    tasks = np.unique(np.asarray(task_ids, np.int64))
    # LIBERO-40 task IDs are arranged in four ten-task suites.  This grouping is
    # only used to sample training tasks, never to inspect held-out evaluation labels.
    suites = [tasks[(tasks // 10) == i] for i in range(4)]
    out = {'all40': tasks.tolist()}
    rng = np.random.default_rng(SEED)
    for size in (20, 10):
        for repeat in range(3):
            picks = []
            quotas = [size // 4 + (i < size % 4) for i in range(4)]
            for suite, quota in zip(suites, quotas):
                if len(suite) < quota:
                    raise ValueError(f'cannot sample {quota} tasks from suite with {len(suite)} tasks')
                picks.extend(rng.choice(suite, quota, replace=False).tolist())
            out[f'{size}tasks_seed{repeat}'] = sorted(picks)
    return out


def pca_task_fit(rows: dict, actions: np.ndarray, dims: int, task_subset: list[int],
                 variable: bool = False):
    mask = np.isin(np.asarray(rows['task']), task_subset) & ~np.asarray(rows['validation'], bool)
    fit = actions[mask]
    if not variable:
        return pca_fit(fit, dims), int(mask.sum())
    traces, _ = stitch(fit, np.stack([rows['task'][mask], rows['episode'][mask]], 1), np.asarray(rows['frame'])[mask])
    phases = [phase_resample(t[s:e, :7]) for t in traces.values()
               for s, e in motion.anchor_spans(t)]
    return pca_fit(np.stack(phases), dims), len(phases)


def run_full(out: Path) -> dict:
    """Full offline comparison. This path is for the experiment owner, not smoke."""
    d = prepare()
    rows = d['rows']
    target, hard = d['eval'], d['hard']
    mem = d['memory']
    train_ids = d['train_ids']
    fixed_rows = []
    fixed_curve = []
    vq_rows = []
    for tau in TAUS:
        unit_ids = greedy_cover(mem, tau)
        reps = mem[unit_ids]
        ix, _ = nearest_templates(target, reps)
        stats = error_summary(reps[ix], target, hard)
        fixed_rows.append({'method': 'raw', 'V': 70, 'tau': tau, 'units': len(unit_ids),
                           'values_per_unit': 70, 'stored_values': int(len(unit_ids) * 70), **stats})
        fixed_curve.append({'representation': 'raw70', 'units': len(unit_ids), 'values_per_unit': 70,
                            'stored_values': int(len(unit_ids) * 70), 'all_mse': stats['all_mse'],
                            'hard_mse': stats['hard_mse'], 'tau': tau})
    for v in VALUE_COUNTS:
        k = v // 7
        for method in ('dct', 'keyframe', 'pca'):
            if method == 'pca':
                mean, vt = pca_fit(d['train'], v)
                rec = pca_restore(pca_project(mem, mean, vt), mean, vt)
            elif method == 'dct':
                rec = dct_decode(dct_encode(mem, k), 10).astype(np.float32)
            else:
                rec = keyframe_decode(keyframe_encode(mem, k), 10).astype(np.float32)
            for tau in TAUS:
                unit_ids = greedy_cover(rec, tau)
                reps = rec[unit_ids]
                ix, _ = nearest_templates(target, reps)
                stats = error_summary(reps[ix], target, hard)
                row = {'method': method, 'V': v, 'tau': tau, 'units': len(unit_ids),
                       'values_per_unit': v, 'stored_values': int(len(unit_ids) * v),
                       **stats}
                fixed_rows.append(row)
                fixed_curve.append({'representation': f'{method}{v}', 'units': len(unit_ids),
                                    'values_per_unit': v, 'stored_values': int(len(unit_ids) * v),
                                    'all_mse': stats['all_mse'], 'hard_mse': stats['hard_mse'], 'tau': tau})
    # VQ: separate one-code-per-unit curve; K is codebook capacity and actual used
    # codes are reported after assigning all memory windows.
    for code_count in (64, 256, 1024):
        km = MiniBatchKMeans(n_clusters=code_count, random_state=SEED, batch_size=2048,
                             n_init=3, max_iter=150)
        km.fit(d['train'].reshape(len(d['train']), -1))
        mem_labels = km.predict(mem.reshape(len(mem), -1))
        used = np.unique(mem_labels)
        centers = km.cluster_centers_[used].reshape(-1, 10, 7)
        qix, _ = nearest_templates(target, centers)
        stats = error_summary(centers[qix], target, hard)
        vq_rows.append({'codebook_capacity': code_count, 'used_codes': len(used),
                        'units': len(used), 'values_per_unit': 1, 'stored_values': int(len(used)),
                        **stats})
    # Existing M1 references are attached, with their known image orientation caveat.
    ref_path = ROOT / 'unseen-libero' / 'draft-reconstruction-01' / 'compression_compare.json'
    ref_data = json.loads(ref_path.read_text()) if ref_path.exists() else None
    m1_refs = []
    if ref_data:
        m1_refs = [r for r in ref_data.get('results', []) if r.get('method') in ('m1b12', 'm1b24')]

    # B: exact anchor segmentation reproduction then variable code families.
    baseline, _ = reproduce_anchor_dct3(d)
    if len(variable_templates(d['memory_traces'])[0]) != BASELINE['units']:
        raise AssertionError('anchor segmentation no longer reproduces 4,516 units')
    var_actions, var_lengths, _ = variable_templates(d['memory_traces'])
    var_rows = []
    for family, dims in [('dct', 2), ('dct', 3), ('dct', 4)]:
        items = compressed_variable(var_actions, family, dims)
        for tau in TAUS:
            unit_ids = cover_variable(items, tau)
            pred, info = score_variable_memory(items, var_lengths, unit_ids, d['eval_traces'], d['rows_of'])
            stats = error_summary(pred, target, hard)
            var_rows.append({'method': f'anchor_variable_dct{dims}', 'tau': tau,
                             'units': len(unit_ids), 'values_per_unit': 7 * dims + 1,
                             'stored_values': int(len(unit_ids) * (7 * dims + 1)), **info, **stats})
    train_segments_all = [tr[s:e, :7] for tr in d['train_traces'].values()
                          for s, e in motion.anchor_spans(tr)]
    for family, dims in [('pca', 14), ('pca', 21), ('pca', 28), ('raw32', 224)]:
        if family == 'raw32':
            mean = basis = None
            stored_per_unit = 32 * 7 + 1
        else:
            mean, basis = pca_fit(np.stack([phase_resample(a, 32) for a in train_segments_all]), dims)
            stored_per_unit = dims + 1
        items = compressed_variable(var_actions, family, dims, mean, basis)
        for tau in TAUS:
            unit_ids = cover_variable(items, tau)
            pred, info = score_variable_memory(items, var_lengths, unit_ids, d['eval_traces'], d['rows_of'])
            stats = error_summary(pred, target, hard)
            var_rows.append({'method': f'anchor_variable_{family}{dims}', 'tau': tau,
                             'units': len(unit_ids), 'values_per_unit': stored_per_unit,
                             'stored_values': int(len(unit_ids) * stored_per_unit), **info, **stats})
    task_subsets = balanced_task_subsets(np.asarray(rows['task'])[train_ids])
    self_rows = []
    # Fixed-window PCA transfer and fixed-basis DCT self-reconstruction.
    for k in (2, 3, 4):
        rec = dct_decode(dct_encode(target, k), 10)
        self_rows.append({'domain': 'fixed10', 'method': f'DCT{k}', 'fit_tasks': 'fixed',
                          **error_summary(rec, target, hard)})
    for subset_name, taskset in task_subsets.items():
        for v in VALUE_COUNTS:
            mean, basis = pca_fit(actions_for_tasks(rows, taskset), v)
            rec = pca_restore(pca_project(target, mean, basis), mean, basis)
            self_rows.append({'domain': 'fixed10', 'method': f'PCA{v}', 'fit_tasks': subset_name,
                              'fit_task_ids': taskset, **error_summary(rec, target, hard)})
    # Variable-domain PCA is fit on anchor-cut LIBERO-40 train units only. Eval
    # remains fixed10 to avoid using unseen actions to determine any cut boundary.
    for subset_name, taskset in task_subsets.items():
        train_trace = selected_train_traces(rows, taskset)
        train_segments = [tr[s:e, :7] for tr in train_trace.values() for s, e in motion.anchor_spans(tr)]
        for v in VALUE_COUNTS:
            mean, basis = pca_fit(np.stack([phase_resample(a) for a in train_segments]), v)
            phase_eval = np.stack([phase_resample(a, 32) for a in target])
            z = pca_project(phase_eval, mean, basis)
            rec_phase = pca_restore(z, mean, basis)
            restored = np.stack([phase_resample(a, 10) for a in rec_phase])
            self_rows.append({'domain': 'variable_basis_transfer_to_fixed10', 'method': f'PCA{v}', 'fit_tasks': subset_name,
                              'fit_task_ids': taskset, **error_summary(restored, target, hard)})

    # D: diagnostics on all LIBERO-40 memory units, with length matched fixed windows.
    variable_items = var_actions
    mean_len = int(np.floor(np.mean(var_lengths) + .5))
    fixed10_items = [tr[s:s+10, :7] for tr in d['memory_traces'].values() for s in range(0, len(tr)-9, 10)]
    fixedmean_items = [tr[s:s+mean_len, :7] for tr in d['memory_traces'].values()
                       for s in range(0, len(tr)-mean_len+1, mean_len)]
    diag_sets = {'variable_anchor': variable_items, 'fixed10': fixed10_items,
                 f'fixed{mean_len}': fixedmean_items}
    d_result = {'mean_anchor_length': float(var_lengths.mean()), 'rounded_mean_length': mean_len,
                'unit_counts': {k: len(v) for k, v in diag_sets.items()}, 'low_frequency': {},
                'cluster_separation': {}, 'cut_alignment': {}}
    for name, items in diag_sets.items():
        d_result['low_frequency'][name] = low_frequency_concentration(items)
    cluster_sets, cluster_sample_units = equal_count_samples(diag_sets)
    for name, items in cluster_sets.items():
        phase = np.stack([phase_resample(a, 32) for a in items])
        d_result['cluster_separation'][name] = [between_within_ratio(phase, k) for k in (32, 95, 256)]
    d_result['cluster_sample_units'] = cluster_sample_units
    d_result['cut_alignment']['variable_anchor'] = cut_alignment(d['memory_traces'], motion.anchor_spans)
    d_result['cut_alignment']['fixed10_reference'] = cut_alignment(d['memory_traces'], None, 10)
    d_result['cut_alignment'][f'fixed{mean_len}_reference'] = cut_alignment(d['memory_traces'], None, mean_len)
    result = {'mode': 'full', 'seed': SEED, 'training_rows': len(d['train']),
              'memory_windows': len(mem), 'eval_windows': len(target), 'hard_windows': int(hard.sum()),
              'A_fixed': fixed_rows, 'A_vq': vq_rows, 'A_M1_reference': m1_refs,
              'B_anchor_baseline': baseline, 'B_variable': var_rows,
              'C_self_reconstruction': self_rows, 'D_memory_variance': d_result,
              'protocol': {'fit_condition': 'LIBERO-40 ~validation only',
                           'eval_action_use': {'A_B': 'nearest-template oracle choice and scoring only',
                                               'C': 'self-encode/decode input and scoring; fitted bases remain LIBERO-40 train only',
                                               'never': 'basis/codebook fitting, segmentation cuts, or merge thresholds'},
                           'variable_cut': 'motion_core_eval.anchor_spans: smoothed physical translational speed local minima + gripper switches, min_length=5',
                           'no_denoising': True, 'task_subsets': task_subsets}}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    csv_rows = fixed_curve + [{'representation': f"vq{r['codebook_capacity']}", 'units': r['units'],
                               'values_per_unit': 1, 'stored_values': r['stored_values'],
                               'all_mse': r['all_mse'], 'hard_mse': r['hard_mse'], 'tau': 'n/a'} for r in vq_rows]
    (out / 'rate_distortion.csv').write_text('representation,units,values_per_unit,stored_values,all_mse,hard_mse,tau\n' + ''.join(
        f"{r['representation']},{r['units']},{r['values_per_unit']},{r['stored_values']},{r['all_mse']},{r['hard_mse']},{r.get('tau')}\n" for r in csv_rows))
    (out / 'report.md').write_text(render_full_report(result))
    return result


def actions_for_tasks(rows: dict, tasks: list[int]) -> np.ndarray:
    mask = np.isin(np.asarray(rows['task']), tasks) & ~np.asarray(rows['validation'], bool)
    return np.asarray(rows['actions'][mask, :, :7], np.float32)


def selected_train_traces(rows: dict, tasks: list[int]) -> dict:
    mask = np.isin(np.asarray(rows['task']), tasks) & ~np.asarray(rows['validation'], bool)
    keys = np.stack([rows['task'][mask], rows['episode'][mask]], 1)
    traces, _ = stitch(np.asarray(rows['actions'][mask, :, :7]), keys, np.asarray(rows['frame'])[mask])
    return traces


def render_full_report(result: dict) -> str:
    lines = ['# CMP2 matched-value compression comparison', '',
             '## Summary', '',
             f"Variable DCT3 anchor baseline reproduced: {result['B_anchor_baseline']['all_mse']:.6f} / {result['B_anchor_baseline']['hard_mse']:.6f} active7 MSE, {result['B_anchor_baseline']['n']} windows.",
             'All bases/codebooks are fit on LIBERO-40 rows with `validation=False`; cuts and merge thresholds use LIBERO-40 memory only. In A/B, unseen LIBERO-90 actions only choose the closest stored draft and score it. C is the declared self-reconstruction exception: an Eval window is encoded/decoded with the already-fitted representation to measure transfer error. No Eval action fits a basis/codebook or determines a cut/merge threshold. No denoising is used.', '',
             '## A. Fixed 10-step windows', '',
             '| Method | V | τ | units | values/unit | stored values | all MSE | hard MSE |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['A_fixed']:
        lines.append(f"| {r['method']} | {r['V']} | {r['tau']} | {r['units']} | {r['values_per_unit']} | {r['stored_values']} | {r['all_mse']:.5f} | {r['hard_mse']:.5f} |")
    lines += ['', 'VQ is separate: each memory item maps to one code index; actual used-code count is reported. M1 b12/b24 rows below are legacy references from `compression_compare.json` and retain its image-orientation caveat (†).', '',
              '| VQ capacity | used units | values/unit | stored values | all MSE | hard MSE |', '|---:|---:|---:|---:|---:|---:|']
    for r in result['A_vq']:
        lines.append(f"| {r['codebook_capacity']} | {r['used_codes']} | 1 | {r['stored_values']} | {r['all_mse']:.5f} | {r['hard_mse']:.5f} |")
    lines += ['', 'Legacy M1 references (†, not refit):', '', '| Method | τ | units | stored values | all MSE | hard MSE |', '|---|---:|---:|---:|---:|---:|']
    for r in result['A_M1_reference']:
        lines.append(f"| {r['method']} | {r['level']} | {r['units']} | {r['stored_values']} | {r['all']:.5f} | {r['hard']:.5f} |")
    lines += ['', '## B. Variable-length anchor units', '',
              'The reproduced baseline uses smoothed physical translational-speed minima plus gripper switches (minimum segment length 5), not the error-threshold segmenter.', '',
              '| Method | τ | units | values/unit | stored values | frames/selection | all MSE | hard MSE |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['B_variable']:
        lines.append(f"| {r['method']} | {r['tau']} | {r['units']} | {r['values_per_unit']} | {r['stored_values']} | {r['frames_per_selection']:.2f} | {r['all_mse']:.5f} | {r['hard_mse']:.5f} |")
    lines += ['', '## C. Self reconstruction on Eval', '',
              '| Domain | Method | Fit subset | all MSE | hard MSE / frame MSE |', '|---|---|---|---:|---:|']
    for r in result['C_self_reconstruction']:
        lines.append(f"| {r['domain']} | {r['method']} | {r['fit_tasks']} | {r.get('all_mse', float('nan')):.5f} | {r.get('hard_mse', r.get('frame_mse', float('nan'))):.5f} |")
    d = result['D_memory_variance']
    lines += ['', '## D. Variable-length variance diagnostics (LIBERO-40 memory only)', '',
              f"Mean anchor length {d['mean_anchor_length']:.2f}; length-matched fixed window {d['rounded_mean_length']}; units: `{d['unit_counts']}`.", '',
              '| Segmentation | Low-frequency DCT energy mean [q25, median, q75] | Cut alignment mean |', '|---|---|---:|']
    for k in d['unit_counts']:
        low, align = d['low_frequency'][k], d['cut_alignment'].get(k)
        if align is None:
            align = d['cut_alignment'].get(f'{k}_reference', {'mean_nearest_normalized_distance': None})
        lines.append(f"| {k} | {low['mean']:.4f} [{low['q25']:.4f}, {low['median']:.4f}, {low['q75']:.4f}] | {align['mean_nearest_normalized_distance']} |")
    lines += ['', f"Cluster-separation ratios use {d['cluster_sample_units']} deterministically sampled units per segmentation; between-type / within-type mean squared distances are in `results.json` under `D_memory_variance.cluster_separation`.", '',
              '## Reproduction command', '', '```bash',
              'cd /work && source env/activate.sh',
              'taskset -c 8-47 nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python skill_moe/skillnet/examples/memory_pilot/unseen_libero/compression_matched.py --full --output /work/.skillnet-state/outputs/memory-pilot/unseen-libero/compress-matched-01-full',
              '```', '']
    return '\n'.join(lines)


def variable_encode_decode(actions: list[np.ndarray], k: int) -> list[np.ndarray]:
    out = []
    for a in actions:
        c = dct_encode(a, min(k, len(a)))
        out.append(dct_decode(c, len(a)))
    return out


def phase_resample(a: np.ndarray, phases: int = 32) -> np.ndarray:
    src = np.linspace(0., 1., len(a))
    dst = np.linspace(0., 1., phases)
    return np.stack([np.interp(dst, src, np.asarray(a)[:, j]) for j in range(a.shape[-1])], -1)


def variable_pca_fit(actions: list[np.ndarray], dims: int):
    phase = np.stack([phase_resample(a) for a in actions])
    return pca_fit(phase, dims)


def phase_pca_restore(a: np.ndarray, mean: np.ndarray, basis: np.ndarray, length: int) -> np.ndarray:
    z = pca_project(phase_resample(a)[None], mean, basis)
    phase = pca_restore(z, mean, basis)[0]
    src, dst = np.linspace(0., 1., len(phase)), np.linspace(0., 1., length)
    return np.stack([np.interp(dst, src, phase[:, j]) for j in range(phase.shape[-1])], -1)


def error_summary(pred: np.ndarray, target: np.ndarray, hard: np.ndarray) -> dict:
    err = np.mean((np.asarray(pred)[..., :7] - np.asarray(target)[..., :7]) ** 2, axis=(1, 2))
    return {'n': int(len(err)), 'hard_n': int(np.asarray(hard).sum()),
            'all_mse': float(err.mean()), 'hard_mse': float(err[np.asarray(hard)].mean()) if np.any(hard) else None,
            'median_mse': float(np.median(err))}


def low_frequency_concentration(actions: list[np.ndarray]) -> dict:
    values = []
    for a in actions:
        x = np.asarray(a, np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        spec = dct(x, type=2, norm='ortho', axis=0)
        energy = np.sum(spec * spec, axis=1)
        denom = float(energy.sum())
        values.append(float(energy[:min(3, len(energy))].sum() / denom) if denom > 1e-20 else 1.)
    q = np.quantile(values, [.25, .5, .75])
    return {'mean': float(np.mean(values)), 'q25': float(q[0]), 'median': float(q[1]), 'q75': float(q[2])}


def between_within_ratio(phase: np.ndarray, clusters: int, seed: int = SEED) -> dict:
    x = np.asarray(phase, np.float32).reshape(len(phase), -1)
    k = min(int(clusters), len(x))
    model = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=min(1024, max(64, k * 4)),
                            n_init=3, max_iter=100, reassignment_ratio=.01)
    labels = model.fit_predict(x)
    centers = model.cluster_centers_
    if len(centers) > 1:
        pair = centers[:, None, :] - centers[None, :, :]
        between = float(np.mean(np.sum(pair * pair, axis=-1)[np.triu_indices(len(centers), 1)]))
    else:
        between = 0.
    residual = x - centers[labels]
    within = float(np.mean(np.sum(residual * residual, axis=1)))
    return {'k_requested': int(clusters), 'k_used': int(k), 'between_center_sqdist': between,
            'within_sqdist': within, 'ratio': float(between / max(within, 1e-20)),
            'occupied_clusters': int(len(np.unique(labels)))}


def cut_alignment(traces: dict, cut_fn, fixed_len: int | None = None) -> dict:
    by_task = {}
    for key, trace in traces.items():
        n = len(trace)
        if fixed_len is None:
            spans = cut_fn(trace)
            points = np.asarray([e / n for _, e in spans[:-1]], np.float64)
        else:
            points = np.arange(fixed_len, n, fixed_len, dtype=np.float64) / n
        by_task.setdefault(int(key[0]), []).append(points)
    diffs = []
    for demos in by_task.values():
        if len(demos) < 2:
            continue
        for i, a in enumerate(demos):
            for j, b in enumerate(demos):
                if i == j or not len(a) or not len(b):
                    continue
                diffs.extend(np.min(np.abs(a[:, None] - b[None, :]), axis=1).tolist())
    return {'pairs': int(len(diffs)), 'mean_nearest_normalized_distance': float(np.mean(diffs)) if diffs else None}


def _row_groups(actions, task, episode, frame, mask):
    keys = np.stack([np.asarray(task)[mask], np.asarray(episode)[mask]], 1)
    return stitch(np.asarray(actions)[mask], keys, np.asarray(frame)[mask])[0]


def prepare():
    """Load existing caches once and return training windows, memory and Eval."""
    stored.load_data()
    S = ev.Shared
    rows = {k: np.load(RUN / 'rows' / f'{k}.npy', mmap_mode='r')
            for k in ('actions', 'task', 'episode', 'frame', 'validation', 'memory')}
    actions = np.asarray(rows['actions'][:, :, :7], np.float32)
    val = np.asarray(rows['validation'], bool)
    train_ids = np.flatnonzero(~val)
    mem_ids = np.flatnonzero(np.asarray(rows['memory'], bool))
    train = actions[train_ids]
    memory = actions[mem_ids]
    train_keys = np.stack([rows['task'][train_ids], rows['episode'][train_ids]], 1)
    train_traces, _ = stitch(actions[train_ids], train_keys, np.asarray(rows['frame'])[train_ids])
    return {'rows': rows, 'train_ids': train_ids, 'mem_ids': mem_ids, 'train': train,
            'memory': memory, 'memory_traces': S.memory, 'eval': np.asarray(S.target, np.float32),
            'hard': np.asarray(S.hard, bool), 'eval_traces': S.traces, 'rows_of': S.rows_of,
            'train_traces': train_traces}


def run_smoke(out: Path, n_eval: int = 8, n_fit: int = 2500, n_diag: int = 500) -> dict:
    d = prepare()
    hard_ids = np.flatnonzero(d['hard'])[:max(1, n_eval // 2)]
    easy_ids = np.flatnonzero(~d['hard'])[:max(0, n_eval - len(hard_ids))]
    eval_ids = np.r_[hard_ids, easy_ids]
    target, hard = d['eval'][eval_ids], d['hard'][eval_ids]
    # Restrict smoke rows at the query only; fitting stays solely on LIBERO-40 train.
    rng = np.random.default_rng(SEED)
    fit_ids = rng.choice(len(d['train']), min(n_fit, len(d['train'])), replace=False)
    fit = d['train'][fit_ids]
    mem = d['memory']
    if len(target) != n_eval:
        raise AssertionError('smoke eval selection unexpectedly short')

    fixed = {}
    fixed_results = []
    raw_ids, _ = nearest_templates(target, mem)
    fixed_results.append({'method': 'raw70_reference', 'requested_values_per_unit': 70,
                          'actual_values_per_unit': 70, 'units': int(len(mem)),
                          'stored_values': int(len(mem) * 70),
                          **error_summary(mem[raw_ids], target, hard)})
    for v in VALUE_COUNTS:
        per_axis = v // 7
        for name, encdec in (
            (f'dct{per_axis}', lambda x, k=per_axis: dct_decode(dct_encode(x, k), 10)),
            (f'keyframe{per_axis}', lambda x, k=per_axis: keyframe_decode(keyframe_encode(x, k), 10)),
        ):
            rec = encdec(mem)
            ix = greedy_cover(rec, None)
            chosen, _ = nearest_templates(target, rec[ix])
            pred = rec[ix[chosen]]
            row = {'method': name, 'requested_values_per_unit': v,
                   'actual_values_per_unit': v, 'units': int(len(ix)), 'stored_values': int(len(ix) * v),
                   **error_summary(pred, target, hard)}
            fixed_results.append(row)
            fixed[name] = rec
        mean, vt = pca_fit(fit, v)
        z = pca_project(mem, mean, vt)
        rec = pca_restore(z, mean, vt)
        ix, _ = nearest_templates(target, rec)
        pred = rec[ix]
        row = {'method': f'pca{v}', 'requested_values_per_unit': v,
               'actual_values_per_unit': v, 'units': int(len(rec)), 'stored_values': int(len(rec) * v),
               'fit_rows': int(len(fit)), **error_summary(pred, target, hard)}
        fixed_results.append(row)
        fixed[f'pca{v}'] = rec

    # Smoke VQ codebooks on a bounded, fixed LIBERO-40 sample; full runs use all train rows.
    vq_results = []
    for k in (64, 256, 1024):
        km = MiniBatchKMeans(n_clusters=min(k, len(fit)), random_state=SEED, batch_size=1024,
                             n_init=1, max_iter=30)
        km.fit(fit.reshape(len(fit), -1))
        centers = km.cluster_centers_.reshape(-1, 10, 7)
        ids = km.predict(mem.reshape(len(mem), -1))
        used = np.unique(ids)
        code_templates = centers[used]
        predix, _ = nearest_templates(target, code_templates)
        row = {'method': f'vq{k}', 'requested_codebook': k, 'used_codes_in_memory': int(len(used)),
               'values_per_unit': 1, 'stored_values': int(len(used)), 'fit_rows': int(len(fit)),
               **error_summary(code_templates[predix], target, hard)}
        vq_results.append(row)

    # Variable representation: use the exact anchor segmentation required by the
    # reference row. Validate count and metrics on all Eval windows before partial runs.
    mem_actions, mem_lengths, mem_keys = variable_templates(d['memory_traces'])
    if len(mem_actions) != BASELINE['units']:
        raise AssertionError(f"anchor segment count mismatch: {len(mem_actions)} != {BASELINE['units']}")
    baseline_prediction, baseline_hard = reproduce_anchor_dct3(d)
    if abs(baseline_prediction['all_mse'] - BASELINE['all']) > 2e-6 or abs(baseline_prediction['hard_mse'] - BASELINE['hard']) > 2e-6:
        raise AssertionError(f'anchor DCT3 baseline mismatch: {baseline_prediction}; expected {BASELINE}')
    variable_results = []
    for k in (2, 3, 4):
        coded = compressed_variable(mem_actions, 'dct', k)
        pred, info = score_variable_memory(coded, mem_lengths, np.arange(len(coded)),
                                           d['eval_traces'], d['rows_of'])
        stats = error_summary(pred[eval_ids], target, hard)
        variable_results.append({'method': f'variable_anchor_dct{k}', 'units': len(coded),
                                 'values_per_unit': 7 * k + 1, 'stored_values': len(coded) * (7 * k + 1),
                                 **info, **stats})

    # C: self reconstruction on a small subset. PCA bases are learned from LIBERO-40 train.
    self_results = []
    for k in (2, 3, 4):
        pred = dct_decode(dct_encode(target, k), 10)
        self_results.append({'method': f'dct{k}', **error_summary(pred, target, hard)})
    for v in VALUE_COUNTS:
        mean, vt = pca_fit(fit, v)
        z = pca_project(target, mean, vt)
        self_results.append({'method': f'pca{v}', 'fit_rows': len(fit),
                             **error_summary(pca_restore(z, mean, vt), target, hard)})
    # No unseen action is used to establish variable segmentation. Instead fit
    # variable-domain PCA on LIBERO-40 train anchor units and evaluate its transfer
    # to the predeclared fixed10 Eval windows after phase resampling.
    variable_fit_segments = [tr[s:e, :7] for tr in d['train_traces'].values()
                             for s, e in motion.anchor_spans(tr)]
    variable_self = []
    for v in VALUE_COUNTS:
        mean, basis = pca_fit(np.stack([phase_resample(a) for a in variable_fit_segments]), v)
        phase_eval = np.stack([phase_resample(a, 32) for a in target])
        z = pca_project(phase_eval, mean, basis)
        phase_rec = pca_restore(z, mean, basis)
        restored = np.stack([phase_resample(a, 10) for a in phase_rec])
        variable_self.append({'method': f'variable_basis_pca{v}_on_fixed10_eval', 'fit_segments': len(variable_fit_segments),
                              **error_summary(restored, target, hard)})

    # D: memory-only sample to keep smoke quick, still no unseen data in these stats.
    diag_actions = mem_actions[:min(n_diag, len(mem_actions))]
    lengths = np.asarray([len(a) for a in diag_actions])
    mean_length = int(np.floor(lengths.mean() + .5))
    fixed10 = [a for tr in list(d['memory_traces'].values())[:20] for a in [tr[i:i+10, :7] for i in range(0, len(tr)-9, 10)]]
    fixedmean = [tr[i:i+mean_length, :7] for tr in list(d['memory_traces'].values())[:20]
                 for i in range(0, len(tr)-mean_length+1, mean_length)]
    d_diag = {'mean_anchor_length': float(lengths.mean()), 'rounded_mean_length': mean_length,
              'sampled_anchor_units': len(diag_actions),
              'low_frequency': {'anchor_variable': low_frequency_concentration(diag_actions),
                                'fixed10': low_frequency_concentration(fixed10),
                                f'fixed{mean_length}': low_frequency_concentration(fixedmean)},
              'cluster_separation': {},
              'cut_alignment': {'anchor_variable': cut_alignment(d['memory_traces'], motion.anchor_spans),
                                'fixed10_reference': cut_alignment(d['memory_traces'], None, 10),
                                f'fixed{mean_length}_reference': cut_alignment(d['memory_traces'], None, mean_length)}}
    diagnostic_groups = {'anchor_variable': diag_actions, 'fixed10': fixed10,
                         f'fixed{mean_length}': fixedmean}
    cluster_groups, cluster_sample_units = equal_count_samples(diagnostic_groups)
    for label, seqs in cluster_groups.items():
        phases = np.stack([phase_resample(a, 32) for a in seqs])
        d_diag['cluster_separation'][label] = [between_within_ratio(phases, k) for k in (32, 95, 256)]
    d_diag['cluster_sample_units'] = cluster_sample_units

    result = {'mode': 'smoke', 'seed': SEED,
              'protocol': {'fit_rows': len(fit), 'fit_source': 'LIBERO-40 rows where validation=False',
                           'memory_units': len(mem), 'scored_eval_windows': len(target),
                           'eval_row_indices': eval_ids.tolist(),
              'eval_source': {'A_B': 'LIBERO-90 Eval; target used for nearest-template selection/scoring',
                              'C': 'LIBERO-90 Eval; target is self-encode/decode input and is scored',
                              'fit_and_cuts': 'no LIBERO-90 Eval actions used'},
                           'hard_threshold': 'inherited from stored_units_eval; smoke hard count may be zero',
                           'vq_note': 'smoke codebooks fitted on bounded train subset; full run uses all train rows'},
              'A_fixed': fixed_results, 'A_vq': vq_results,
              'B_anchor_reproduction': {'actual_units': len(mem_actions), 'expected': BASELINE,
                                        'reproduced': baseline_prediction, 'passed': True,
                                        'variable_dct_smoke': variable_results},
              'C_self_reconstruction': {'fixed_eval_subset': self_results, 'variable_eval_subset': variable_self},
              'D_variance_diagnostics_smoke': d_diag}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    (out / 'rate_distortion.csv').write_text('representation,units,values_per_unit,stored_values,all_mse,hard_mse,scope\n' + ''.join(
        f"{r['method']},{r['units']},{r['actual_values_per_unit']},{r['stored_values']},{r['all_mse']},{r['hard_mse']},smoke_eval\n" for r in fixed_results))
    (out / 'report.md').write_text(render_report(result, out))
    return result


def reproduce_anchor_dct3(d: dict) -> tuple[dict, np.ndarray]:
    """Reproduce motion_core_eval anchor/world/joint DCT3 with 10-frame budget."""
    values, lengths = [], []
    for key, trace in d['memory_traces'].items():
        for start, end in motion.anchor_spans(trace):
            values.append(motion.encode(trace, None, start, end, 'world', 1))
            lengths.append(end - start)
    values, lengths = np.stack(values), np.asarray(lengths)
    drafts = motion.drafts_by_length(values, lengths, False, 1)
    keys = sorted(drafts)
    pred = np.empty_like(ev.Shared.target, dtype=np.float64)
    for key, trace in d['eval_traces'].items():
        table = motion.cost_table(trace, {}, drafts, 'world', np.arange(7))
        _, seq = motion.opt.best_sequence(table, keys, len(trace), max(1, math.ceil(len(trace) / 10)))
        joined = np.empty_like(trace, dtype=np.float64)
        for start, length in seq:
            piece = table[start, length][1]
            joined[start:start + len(piece)] = piece
        pred[d['rows_of'][key]] = joined.reshape(-1, 10, 7)
    stats = error_summary(pred, ev.Shared.target, ev.Shared.hard)
    return stats, pred


def render_report(d: dict, output: Path) -> str:
    b = d['B_anchor_reproduction']['reproduced']
    fixed = '\n'.join(f"| {r['method']} | {r['units']} | {r['actual_values_per_unit']} | {r['stored_values']} | {r['all_mse']:.5f} | {r['hard_mse'] if r['hard_mse'] is not None else 'n/a'} |"
                       for r in d['A_fixed'])
    vq = '\n'.join(f"| {r['method']} | {r['used_codes_in_memory']} | 1 | {r['stored_values']} | {r['all_mse']:.5f} | {r['hard_mse'] if r['hard_mse'] is not None else 'n/a'} |"
                   for r in d['A_vq'])
    self_rows = '\n'.join(f"| {r['method']} | {r['all_mse']:.5f} | {r['hard_mse'] if r['hard_mse'] is not None else 'n/a'} |"
                          for r in d['C_self_reconstruction']['fixed_eval_subset'])
    variable_self = '\n'.join(f"| {r['method']} | {r['all_mse']:.5f} | {r['hard_mse'] if r['hard_mse'] is not None else 'n/a'} |"
                               for r in d['C_self_reconstruction']['variable_eval_subset'])
    low = '\n'.join(f"| {name} | {v['mean']:.4f} | {v['median']:.4f} | {v['q25']:.4f}–{v['q75']:.4f} |"
                    for name, v in d['D_variance_diagnostics_smoke']['low_frequency'].items())
    variable = '\n'.join(f"| {r['method']} | {r['units']} | {r['values_per_unit']} | {r['stored_values']} | {r['all_mse']:.5f} | {r['hard_mse'] if r['hard_mse'] is not None else 'n/a'} |"
                         for r in d['B_anchor_reproduction']['variable_dct_smoke'])
    ratios = '\n'.join(f"| {name} | {r['k_used']} | {r['ratio']:.4f} |"
                       for name, rows in d['D_variance_diagnostics_smoke']['cluster_separation'].items()
                       for r in rows)
    return f"""# CMP2 compression matched-value smoke

## B baseline gate

Matched the required speed-minimum + gripper-switch segmentation: **{d['B_anchor_reproduction']['actual_units']:,} units**; DCT3 active7 MSE `{b['all_mse']:.8f}` all / `{b['hard_mse']:.8f}` hard. Expected `.03444790 / .06845424`. This is an exact full-Eval reproduction check, before subset smoke analyses.

## Scope

This output is a smoke run, not a full comparison. PCA and VQ fits used only a deterministic subset of LIBERO-40 rows with `validation=False`; full CMP2 must fit on all such rows. In A/B, LIBERO-90 target actions were used only for template selection and scoring. C intentionally uses each Eval window as input to its own encode/decode self-reconstruction. Eval actions never fit compression bases or codebooks and never determine segment boundaries or merge thresholds. No denoising or search model was used.

### A. Fixed 10-step windows

`results.json` contains smoke rows for DCT, PCA, keyframes and VQ. The full run must evaluate τ = none/.04/.08/.16 and report unit count, values/unit, total stored values, all/hard MSE. VQ is reported separately because each unit stores one code index; codebook size is not the used-unit count.

| Representation | units | values/unit | stored values | all MSE | hard MSE |
|---|---:|---:|---:|---:|---:|
{fixed}

VQ smoke (bounded training subset):

| codebook | used units | values/unit | stored values | all MSE | hard MSE |
|---|---:|---:|---:|---:|---:|
{vq}

### B. Variable units

The validated reference is `motion_core_eval.anchor_spans`: speed minima of smoothed physical translational action plus gripper switches, with minimum span length 5. This differs from reconstruction-error segmentation (`segment_demo.cut`). The variable smoke includes a full-Eval reproduction gate and small-query DCT2/3/4 checks.

| Representation | units | values/unit | stored values | all MSE | hard MSE |
|---|---:|---:|---:|---:|---:|
{variable}

### C. Self reconstruction

PCA fitting is LIBERO-40 train only. To avoid deriving variable cut boundaries from unseen actions, variable-domain PCA is transferred to the predeclared fixed-10 Eval windows; this is not Eval variable-segment reconstruction.

| Fixed10 representation | all MSE | hard MSE |
|---|---:|---:|
{self_rows}

| Variable-domain PCA transferred to fixed10 | all MSE | hard MSE |
|---|---:|---:|
{variable_self}

### D. Memory-only variance diagnostics

The smoke diagnostics use only LIBERO-40 memory actions. They include low-frequency DCT energy, between-center/within-cluster distance ratios, and normalized cut alignment. Values are sample diagnostics; full run should use all memory units.

| Segmentation | low-frequency mean | median | IQR |
|---|---:|---:|---:|
{low}

| Segmentation | k | between/within ratio |
|---|---:|---:|
{ratios}

The D cluster-separation ratios use the same deterministic number of units for each segmentation (`{d['D_variance_diagnostics_smoke']['cluster_sample_units']}` in this smoke). The full run does the same and reports the common count in `results.json`.

Cut-alignment distances are in `results.json`.

## Reproduction command

```bash
cd /work && source env/activate.sh
taskset -c 8-47 nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python skill_moe/skillnet/examples/memory_pilot/unseen_libero/compression_matched.py --smoke --output {output}
```

"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true', help='Run bounded diagnostic smoke only (no full grid).')
    ap.add_argument('--full', action='store_true', help='Run the full CMP2 offline grid.')
    ap.add_argument('--output', type=Path, default=OUT_DEFAULT)
    ap.add_argument('--eval-rows', type=int, default=8)
    ap.add_argument('--fit-rows', type=int, default=2500)
    args = ap.parse_args()
    if args.smoke == args.full:
        raise SystemExit('choose exactly one of --smoke or --full')
    if args.full:
        out = run_full(args.output)
        print(json.dumps({'output': str(args.output), 'anchor_reproduction': out['B_anchor_baseline']}, indent=2))
    else:
        out = run_smoke(args.output, args.eval_rows, args.fit_rows)
        print(json.dumps({'output': str(args.output), 'anchor_reproduction': out['B_anchor_reproduction']['reproduced']}, indent=2))


if __name__ == '__main__':
    main()
