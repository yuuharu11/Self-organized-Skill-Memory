#!/usr/bin/env python3
"""Aggregate resumable T0X runs and write the progress report/results."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from t0x_common import OUT, ROOT, timestamp, write_json

HERE = Path(__file__).resolve().parent
sys_path = str(HERE)
import sys
sys.path.insert(0, sys_path)
import run_ex1_scope_changed as EX1


def read_jsonl(path):
    out = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try: out.append(json.loads(line))
            except json.JSONDecodeError: continue
    return out


def by_trial(rows):
    return {(int(r['task_id']), int(r['trial_index'])): r for r in rows if r.get('status') == 'complete'}


def mcnemar(test, reference):
    keys = sorted(set(test) & set(reference))
    test_only = sum(bool(test[k]['success']) and not bool(reference[k]['success']) for k in keys)
    ref_only = sum(bool(reference[k]['success']) and not bool(test[k]['success']) for k in keys)
    n = test_only + ref_only
    p = 1.0 if n == 0 else min(1.0, 2. * sum(math.comb(n, i) for i in range(min(test_only, ref_only) + 1)) / 2**n)
    return {'pairs': len(keys), 'test_only': int(test_only), 'reference_only': int(ref_only), 'p_exact': p}


def baseline_for_suite(suite):
    tasks, rows = EX1.load_baseline(suite)
    baseline = by_trial(list(rows.values()))
    if suite == 'libero10':
        # KQ2-CL completed trials 5..9 with the same normal pi0.5 client; trials 0..4 are EX1 baseline.
        more = read_jsonl(ROOT / 'kq2-cl-01/closed-loop/libero10/pi05/trials.jsonl')
        for r in more:
            if r.get('status') == 'complete' and int(r['trial_index']) in range(5,10):
                baseline[(int(r['task_id']), int(r['trial_index']))] = r
        if len(baseline) != 100:
            raise AssertionError(f'LIBERO-10 paired pi0.5 reference has {len(baseline)}/100 rows')
    return tasks, baseline


def wilson(k, n):
    if n == 0: return [None, None]
    z = 1.959963984540054; p = k/n; den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0., center-half), min(1., center+half)]


def qsummary(values):
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    return {} if not len(x) else {'p05': float(np.quantile(x,.05)), 'median': float(np.median(x)),
                                  'p95': float(np.quantile(x,.95)), 'mean': float(x.mean())}


def query_stats(rows):
    qs = [q for r in rows for q in r.get('selected_units_and_t0', []) if isinstance(q, dict)]
    fallback = [bool(q.get('fallback_pi05')) for q in qs]
    return {'query_count': len(qs), 'fallback_rate': float(np.mean(fallback)) if fallback else None,
            't0_all': qsummary([q.get('t0') for q in qs]),
            't0_non_fallback': qsummary([q.get('t0') for q in qs if not q.get('fallback_pi05')]),
            'ehat': qsummary([q.get('confidence_error') for q in qs])}


def switches(row):
    commands = []
    for q in row.get('selected_units_and_t0', []):
        if isinstance(q, dict): commands.extend(np.sign(q.get('gripper_chunk', [])[:5]).tolist())
    return int(sum(a != b for a,b in zip(commands, commands[1:])))


def summarize_pair(stage, suite, condition, refs):
    path = OUT / stage / 'closed-loop' / suite / condition / 'trials.jsonl'
    raw = read_jsonl(path); rows = by_trial(raw)
    tasks, baseline = baseline_for_suite(suite)
    score = {'completed': len(rows), 'successes': int(sum(bool(r['success']) for r in rows.values())),
             'wilson95': wilson(sum(bool(r['success']) for r in rows.values()), len(rows)),
             'tasks': {str(t['task_id']): {'name': t['name'], 'successes': int(sum(bool(rows.get((t['task_id'],i),{}).get('success',False)) for i in range(10))),
                                                  'completed': sum((t['task_id'],i) in rows for i in range(10))} for t in tasks},
             'query_diagnostics': query_stats(list(rows.values()))}
    score['mcnemar_vs_pi05'] = mcnemar(rows, baseline)
    if 'Aconst' in refs:
        score['mcnemar_vs_Aconst'] = mcnemar(rows, refs['Aconst'])
    score['median_success_steps'] = float(np.median([r['env_steps'] for r in rows.values() if r['success']])) if any(r['success'] for r in rows.values()) else None
    return score


def t3_failure_diagnostic():
    p = ROOT / 'kq2-cl-01/closed-loop/libero10/Aconst/trials.jsonl'
    rows = [r for r in read_jsonl(p) if r.get('status') == 'complete' and not r.get('success')]
    manifest = json.loads((ROOT.parent / 'pi05-memory40-20261104/run-01/data_manifest.json').read_text())
    task_names = {int(k): ' '.join(v.lower().replace('_',' ').split()) for k,v in manifest['tasks'].items()}
    all_q = [q for r in rows for q in r.get('selected_units_and_t0', []) if isinstance(q, dict)]
    matching, unknown = [], 0
    for r in rows:
        target = ' '.join(r['task'].lower().replace('_',' ').split())
        for q in r.get('selected_units_and_t0', []):
            if not isinstance(q, dict) or 'unit_task' not in q: continue
            unit_name = task_names.get(int(q['unit_task']))
            if unit_name is None: unknown += 1
            else: matching.append(unit_name == target)
    term = {}
    for r in rows: term[r.get('termination_reason','unknown')] = term.get(r.get('termination_reason','unknown'), 0) + 1
    return {'source': str(p), 'failed_trials': len(rows), 'termination_reasons': term,
            'env_steps': qsummary([r.get('env_steps') for r in rows]),
            'queries_per_trial': qsummary([r.get('num_policy_queries') for r in rows]),
            'fallback_rate': float(np.mean([bool(q.get('fallback_pi05')) for q in all_q])) if all_q else None,
            'fallback_rate_per_trial': qsummary([query_stats([r])['fallback_rate'] for r in rows]),
            'gripper_switches_per_trial': qsummary([switches(r) for r in rows]),
            'unit_task_same_instruction_rate': float(np.mean(matching)) if matching else None,
            'unit_task_comparable_selections': len(matching), 'unit_task_unmapped_selections': unknown,
            'stage_progress_note': 'Trial records do not encode semantic subtask stages. All failures can be located by termination reason and control-step/query count; manual video stage labels are unavailable in these records.'}


def build():
    result = {'created_jst': timestamp(), 'status': {}, 't1': {}, 't3': {}, 't2': {},
              'smoke': {}, 'prior_kq2cl': {'libero90_A': 28, 'libero90_Aconst': 35, 'libero90_B': 7,
                                              'libero90_Bconst': 32, 'libero90_pi05': 32,
                                              'libero10_A': 25, 'libero10_Aconst': 68,
                                              'libero10_B': 3, 'libero10_pi05': 95}}
    # Store T1 success and paired references for each condition/suite.
    for suite in ('libero90', 'libero10'):
        refs = {}
        ap = ROOT / 'kq2-cl-01/closed-loop' / suite / 'Aconst/trials.jsonl'
        refs['Aconst'] = by_trial(read_jsonl(ap))
        for cond in ('Aadd','Badd'):
            s = summarize_pair('t1', suite, cond, refs)
            result['t1'].setdefault(suite,{})[cond] = s
        result['status'][f't1_{suite}'] = {c:v['completed'] for c,v in result['t1'][suite].items()}
    # T3 comparisons are L10 only.
    refs = {'Aconst': by_trial(read_jsonl(ROOT / 'kq2-cl-01/closed-loop/libero10/Aconst/trials.jsonl'))}
    for cond in ('AconstPG','AconstT25'):
        s = summarize_pair('t3','libero10',cond,refs)
        result['t3'].setdefault('libero10',{})[cond] = s
    result['t3']['Aconst_L10_failures'] = t3_failure_diagnostic()
    # Smoke snapshot.
    sr = read_jsonl(OUT / 'smoke/closed-loop-smoke/Aadd/trials.jsonl')
    result['smoke']['Aadd_libero90_task10'] = {'completed': len([r for r in sr if r.get('status') == 'complete']),
        'successes': sum(bool(r.get('success')) for r in sr if r.get('status') == 'complete'),
        'query_diagnostics': query_stats([r for r in sr if r.get('status') == 'complete'])}
    spg = read_jsonl(OUT / 'smoke/closed-loop-smoke-libero10/AconstPG/trials.jsonl')
    result['smoke']['AconstPG_libero10_task0'] = {
        'completed': len([r for r in spg if r.get('status') == 'complete']),
        'successes': sum(bool(r.get('success')) for r in spg if r.get('status') == 'complete'),
        'query_diagnostics': query_stats([r for r in spg if r.get('status') == 'complete'])}
    # T2 correlations determine whether A-exp is permitted.
    d = OUT / 't2/displacement.json'
    if d.exists():
        disp = json.loads(d.read_text()); result['t2'] = disp
        dev = disp.get('splits',{}).get('dev2',{}).get('iii',{})
        gains = {k: float(dev.get(k,{}).get('m_spearman',float('nan')) - dev.get('ehat_spearman',float('nan')))
                 for k in ('t0.15','t0.30') if k in dev}
        best_key = max(gains,key=gains.get) if gains else None
        result['t2']['dev_gain_vs_ehat'] = gains
        result['t2']['Aexp_gate_pass'] = bool(best_key and gains[best_key] >= .10)
        result['t2']['selected_probe_for_Aexp'] = best_key if result['t2']['Aexp_gate_pass'] else None
    result['status']['Aexp_gate'] = result['t2'].get('Aexp_gate_pass',False)
    return result


def md_report(r):
    l90 = r.get('t1', {}).get('libero90', {})
    l10 = r.get('t1', {}).get('libero10', {})
    l90a = l90.get('Aadd', {}).get('successes')
    l90b = l90.get('Badd', {}).get('successes')
    l10a = l10.get('Aadd', {}).get('successes')
    l10b = l10.get('Badd', {}).get('successes')
    deviii = r.get('t2', {}).get('splits', {}).get('dev2', {}).get('iii', {})
    t2_sentence = ('T2 の Dev 修正量 m は ê を +0.1 上回らず、A-exp は実施しない。'
                   if not r.get('t2', {}).get('Aexp_gate_pass') else
                   'T2 の Dev 基準を満たしたため A-exp も評価した。')
    t3vals = r.get('t3', {}).get('libero10', {})
    t3_sentence = ('T3 は A-const-pg={}/100、A-const-t25={}/100。'.format(
        t3vals['AconstPG']['successes'], t3vals['AconstT25']['successes'])
        if all(c in t3vals and t3vals[c]['completed'] == 100 for c in ('AconstPG', 'AconstT25'))
        else 'T3 の閉ループ診断は未完了。')
    conclusion = (f"T1 の加法半径は L90 で A/B とも {l90a if l90a is not None else '—'}/100、"
                  f"L10 では A={l10a if l10a is not None else '—'}/100・B={l10b if l10b is not None else '—'}/100。"
                  f"（L90 B={l90b if l90b is not None else '—'}/100。）" +
                  (f"T2 Dev iii: ê={deviii.get('ehat_spearman', float('nan')):.3f}, "
                   f"m@.15={deviii.get('t0.15', {}).get('m_spearman', float('nan')):.3f}, "
                   f"m@.30={deviii.get('t0.30', {}).get('m_spearman', float('nan')):.3f}; " if deviii else '') +
                  t2_sentence + ' ' + t3_sentence)
    lines = ['# T0X：信頼度を分けた t0 の閉ループ検証', '',
             f'更新時刻: {r["created_jst"]} (JST)', '',
             '## 結論と主表', '',
             conclusion, '',
             'T1 は `t0=clip(sqrt(.15²+c·ê), .15, .35)`、ê>b では π0.5 fallback。T2 は Expert の修正量 m と ê を比較し、Dev Spearman が ê より0.1以上良い場合のみ A-exp を許可する。T3 は L10 の失敗診断と、グリッパーπ0.5差し替え／t0=.25の記述的比較。', '',
             '| Suite | Condition | 成功 | 完了 | Wilson 95% | π0.5のみ / 条件のみ (p; n paired) | A-constのみ / 条件のみ (p; n paired) | fallback | t0 P05/50/95 (非fallback) | ê P05/50/95 |',
             '|---|---|---:|---:|---|---|---|---:|---|---|']
    for suite, conds in r['t1'].items():
        for cond, x in conds.items():
            d=x['query_diagnostics']; mp=x['mcnemar_vs_pi05']; ma=x.get('mcnemar_vs_Aconst',{})
            lines.append(f"| {suite} | {cond} | {x['successes']} | {x['completed']} | {fmt_ci(x['wilson95'])} | {mp['reference_only']} / {mp['test_only']} (p={mp['p_exact']:.3g}; n={mp['pairs']}) | {ma.get('reference_only','—')} / {ma.get('test_only','—')} (p={ma.get('p_exact',float('nan')):.3g}; n={ma.get('pairs','—')}) | {fmt_pct(d['fallback_rate'])} | {fmt_triplet(d['t0_non_fallback'])} | {fmt_triplet(d['ehat'])} |")
    lines += ['', '**タスク別成功数**', '', '| Suite | Condition | Task | 成功/試行 |', '|---|---|---|---:|']
    for suite, conds in r['t1'].items():
        for cond, x in conds.items():
            for tid, task in x['tasks'].items():
                lines.append(f"| {suite} | {cond} | {task['name']} (`{tid}`) | {task['successes']}/{task['completed']} |")
    lines += ['', '条件の対比較は KQ2-CL と同じ初期状態・seed（trial i: env seed 7+i）、wait=10、replan=5、最大520制御step。A-add は query (iii)、B-add は query (i)。各100試行。T1の状態は `results.json` に保存する。', '', '## T2：Expert 修正量による信頼度', '']
    if r.get('t2',{}).get('splits'):
        lines += ['| split | query | ê Spearman | m@.15 Spearman (gain) | m@.30 Spearman (gain) |', '|---|---|---:|---:|---:|']
        for split, qs in r['t2']['splits'].items():
            for q, v in qs.items():
                e=v.get('ehat_spearman',float('nan')); x=v.get('t0.15',{}).get('m_spearman',float('nan')); y=v.get('t0.30',{}).get('m_spearman',float('nan'))
                lines.append(f'| {split} | {q} | {e:.3f} | {x:.3f} ({x-e:+.3f}) | {y:.3f} ({y-e:+.3f}) |')
        lines += ['', f"Dev gate A-exp: {'通過' if r['t2'].get('Aexp_gate_pass') else '不通過'}（基準 +0.10）。Probe: {r['t2'].get('selected_probe_for_Aexp') or '該当なし'}。", '']
    else:
        lines += ['未完了。初回 batch 8 と batch 2 は T1 の2サーバー稼働中に OOM。batch 1 で再実行する。', '']
    lines += ['## T3：L10 診断条件と既存 A-const 失敗', '',
              '| Condition | 成功 | 完了 | Wilson 95% | π0.5のみ / 条件のみ (p; n paired) | A-constのみ / 条件のみ (p; n paired) | fallback | t0 P05/50/95 |',
              '|---|---:|---:|---|---|---|---:|---|']
    for cond, x in r['t3'].get('libero10',{}).items():
        mp=x['mcnemar_vs_pi05']; ma=x.get('mcnemar_vs_Aconst',{}); d=x['query_diagnostics']
        lines.append(f"| {cond} | {x['successes']} | {x['completed']} | {fmt_ci(x['wilson95'])} | {mp['reference_only']} / {mp['test_only']} (p={mp['p_exact']:.3g}; n={mp['pairs']}) | {ma.get('reference_only','—')} / {ma.get('test_only','—')} (p={ma.get('p_exact',float('nan')):.3g}; n={ma.get('pairs','—')}) | {fmt_pct(d['fallback_rate'])} | {fmt_triplet(d['t0_all'])} |")
    lines += ['', '**T3 タスク別成功数**', '', '| Condition | Task | 成功/試行 |', '|---|---|---:|']
    for cond, x in r['t3'].get('libero10', {}).items():
        for tid, task in x['tasks'].items():
            lines.append(f"| {cond} | {task['name']} (`{tid}`) | {task['successes']}/{task['completed']} |")
    f=r['t3'].get('Aconst_L10_failures',{})
    lines += ['', 'A-const-pg はグリッパーだけを π0.5 出力に置換した診断条件であり、提案方式ではない。A-const-t25 は固定 t0=.25。', f"A-const の L10 失敗: {f.get('failed_trials','—')}試行、終了理由 `{f.get('termination_reasons',{})}`。env step中央値 {f.get('env_steps',{}).get('median','—')}、query数中央値 {f.get('queries_per_trial',{}).get('median','—')}、fallback率 {fmt_pct(f.get('fallback_rate'))}、実行グリッパー切替中央値 {f.get('gripper_switches_per_trial',{}).get('median','—')}。", f"選択単位が指示文と一致した割合: {fmt_pct(f.get('unit_task_same_instruction_rate'))}（比較可能な選択 {f.get('unit_task_comparable_selections','—')}、対応付け不明 {f.get('unit_task_unmapped_selections','—')}）。閉ループログからは意味的な作業段階を記録できないため、停止段階の手動分類はできていない。", '',
              '## smoke・実行記録・注意点', '',
              f"A-add smoke (LIBERO-90 task 10, trial 0-1): {r['smoke'].get('Aadd_libero90_task10',{}).get('successes','—')}/2。A-const-pg smoke (LIBERO-10 task 0, trial 0-1): {r['smoke'].get('AconstPG_libero10_task0',{}).get('successes','—')}/2。query trace は `results.json` の smoke 節。", '',
              'KQ2-CL の既存値：L90 A=28/100、A-const=.15=35/100、B=7/100、B-const=32/100、π0.5=32/100。L10 A=25/100、A-const=68/100、B=3/100、π0.5=95/100。', '',
              'GPU OOM: T1 後の空き確認で、T2 は JAX 割合 .15・batch 8/2/1 の順に OOM。GPU が空いていることを再確認し、.40・batch 1 で完了した。稼働中ジョブを止めず、CPU/GPU の thread・fraction を記録した。T1 の技術エラーは `progress.log` と試行 JSONL に記録。', '',
              '## 再現コマンド', '',
              '```bash', 'cd /work && source env/activate.sh',
              'export SKILLNET_STATE_DIR=/work/.skillnet-state XLA_PYTHON_CLIENT_PREALLOCATE=false',
              'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 taskset -c 16-23 python skill_moe/skillnet/examples/memory_pilot/unseen_libero/t0x_run.py t1 --workers 6 --fraction .20 --cpus 24-31,32-39',
              'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 taskset -c 16-23 python skill_moe/skillnet/examples/memory_pilot/unseen_libero/t0x_run.py t3 --workers 6 --fraction .20 --cpus 24-31,32-39',
              'OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_PYTHON_CLIENT_MEM_FRACTION=.40 nice -n 10 taskset -c 8-15 python skill_moe/skillnet/examples/memory_pilot/unseen_libero/t0x_run.py t2 --batch 1 --fraction .40',
              '```', '']
    return '\n'.join(lines)


def fmt_ci(x): return '—' if not x or x[0] is None else f'[{x[0]:.3f}, {x[1]:.3f}]'
def fmt_pct(x): return '—' if x is None else f'{100*x:.1f}%'
def fmt_triplet(x):
    if not x: return '—'
    return '/'.join('—' if x.get(k) is None else f"{x[k]:.3f}" for k in ('p05', 'median', 'p95'))


def main():
    r=build()
    write_json(OUT/'results.json', r)
    (OUT/'report.md').write_text(md_report(r), encoding='utf-8')
    print(json.dumps({'status':r['status'], 'report':str(OUT/'report.md')}, indent=2))


if __name__=='__main__': main()

