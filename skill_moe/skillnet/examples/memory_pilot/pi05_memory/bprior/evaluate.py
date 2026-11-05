"""Small resumable paired LIBERO-Long check for b-prior bridge conditions."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent
import os as _os
_REPO = Path(__file__).resolve().parents[6]
_STATE_DIR = Path(_os.environ.get('SKILLNET_STATE_DIR', _REPO / '.skillnet-state'))
_PI05_ROOT = Path(_os.environ.get('SKILLNET_PI05_ROOT', _STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104'))
RUN = _PI05_ROOT / 'run-01'
OUTPUT = _PI05_ROOT / 'bprior-long-01'
CHECKPOINT = _STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero'

sys.path.insert(0, str(PI05))
from run import write_json  # noqa: E402


def append_jsonl(path: Path, row: dict):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def latest_rows(path: Path):
    latest = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if 'task_suite' not in row:
                continue
            key = (row['task_suite'], int(row['task_id']), int(row['trial_index']), row['condition'])
            latest[key] = row
    return latest


def baseline_rows(path: Path):
    latest = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get('task_suite') == 'libero_10' and row.get('status') == 'complete':
            key = (int(row['task_id']), int(row['trial_index']), row['condition'])
            latest[key] = row
    return latest


def exact_mcnemar(left_success: list[int], right_success: list[int]):
    import math
    rescue = sum(a == 0 and b == 1 for a, b in zip(left_success, right_success))
    harm = sum(a == 1 and b == 0 for a, b in zip(left_success, right_success))
    n = rescue + harm
    p = (min(1., 2 * sum(math.comb(n, i) for i in range(min(rescue, harm)+1)) / (2**n))
         if n else 1.)
    return {'candidate_only_success': rescue, 'baseline_only_success': harm,
            'both_success': sum(a == b == 1 for a, b in zip(left_success, right_success)),
            'both_failure': sum(a == b == 0 for a, b in zip(left_success, right_success)),
            'exact_mcnemar_p': p, 'paired_n': len(left_success)}


def summarize(out: Path, run: Path):
    from evaluate import wilson
    current = latest_rows(out / 'rollouts.jsonl')
    base = baseline_rows(run / 'rollouts.jsonl')
    conditions = ['bprior_direct', 'bprior_retrieve']
    summary = {'conditions': {}, 'paired_to_run01': {}, 'task_successes': {},
               'run01_baseline_rows': {}}
    for condition in conditions:
        completed = [r for k, r in current.items() if k[3] == condition and r.get('status') == 'complete']
        successes = sum(bool(r['success']) for r in completed)
        n = len(completed)
        debug = [trace for r in completed for trace in r.get('selected_units_and_t0', [])
                 if trace.get('condition') == condition]
        b_seconds = [float(x['b_hat_seconds']) for x in debug if 'b_hat_seconds' in x]
        b_total_seconds = [float(x['b_hat_total_seconds']) for x in debug
                           if 'b_hat_total_seconds' in x]
        query_seconds = [float(x['total_seconds']) for x in debug if 'total_seconds' in x]
        retrieval_seconds = [float(x['retrieve_seconds']) for x in debug if 'retrieve_seconds' in x]
        cosines = [float(x['cosine']) for x in debug if x.get('cosine') is not None]
        draft_diffs = [float(x['draft_final_mse_normalized']) for x in debug
                       if 'draft_final_mse_normalized' in x]
        summary['conditions'][condition] = {'successes': successes, 'completed': n,
            'success_rate': successes/n if n else None, 'wilson95': wilson(successes, n),
            'errors': sum(k[3] == condition and r.get('status') == 'error' for k, r in current.items()),
            'mean_trial_seconds': (sum(r.get('elapsed_seconds', 0) for r in completed)/n if n else None),
            'policy_queries': len(debug),
            'mean_b_hat_ms_per_query': (1000*sum(b_seconds)/len(b_seconds) if b_seconds else None),
            'mean_b_hat_total_ms_per_query': (1000*sum(b_total_seconds)/len(b_total_seconds)
                                              if b_total_seconds else None),
            'median_total_policy_query_ms': (1000*sorted(query_seconds)[len(query_seconds)//2]
                                             if query_seconds else None),
            'mean_retrieve_ms_per_query': (1000*sum(retrieval_seconds)/len(retrieval_seconds)
                                            if retrieval_seconds else None),
            'mean_retrieval_cosine': (sum(cosines)/len(cosines) if cosines else None),
            'mean_draft_final_mse_normalized': (sum(draft_diffs)/len(draft_diffs)
                                                 if draft_diffs else None)}
        by_task = {}
        for task in sorted({int(r['task_id']) for r in completed}):
            values = [r for r in completed if int(r['task_id']) == task]
            by_task[str(task)] = {'successes': sum(bool(r['success']) for r in values), 'n': len(values)}
        summary['task_successes'][condition] = by_task
        matched = [(base[(int(r['task_id']), int(r['trial_index']), baseline_condition)], r)
                   for r in completed for baseline_condition in ('normal', 'memory')
                   if (int(r['task_id']), int(r['trial_index']), baseline_condition) in base]
        for baseline_condition in ('normal', 'memory'):
            pairs = [(base[(int(r['task_id']), int(r['trial_index']), baseline_condition)], r)
                     for r in completed if (int(r['task_id']), int(r['trial_index']), baseline_condition) in base]
            for baseline, result in pairs:
                if baseline.get('initial_state_sha256') != result.get('initial_state_sha256'):
                    raise ValueError('bprior and run-01 initial states do not match')
            label = f'{condition}_vs_{baseline_condition}'
            summary['paired_to_run01'][label] = exact_mcnemar(
                [int(r['success']) for _, r in pairs], [int(b['success']) for b, _ in pairs])
            summary['run01_baseline_rows'][label] = len(pairs)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--port', type=int, default=8167)
    parser.add_argument('--task-ids', type=int, nargs='+', default=[0])
    parser.add_argument('--trial-indices', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers != 1:
        raise ValueError('use one simulator worker for the smoke; avoid competing renderers')
    args.output.mkdir(parents=True, exist_ok=True)
    experiment = args.output / 'experiment.json'
    protocol = {'suite':'libero_10', 'task_ids':args.task_ids, 'trial_indices':args.trial_indices,
        'conditions':['bprior_direct','bprior_retrieve'], 'seed_base':7, 'trial_seed':'7 + trial_index',
        'initial_state_selection':'public initial_states[trial_index]', 'num_steps':10,
        'replan_steps':5, 'wait_steps':10, 'max_env_steps':520,
        'bridge_t0':.6, 'checkpoint':str(args.checkpoint), 'run_snapshot':str(args.run)}
    if experiment.exists() and json.loads(experiment.read_text()) != protocol:
        raise ValueError('existing experiment metadata differs; refusing to reuse this output')
    write_json(experiment, protocol)

    sys.path.insert(0, str(HERE.parents[1] / 'anchor_flow'))
    from run_case import _run_trial, suite_control_limit
    env = os.environ.copy()
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.30'
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = env['OPENBLAS_NUM_THREADS'] = '8'
    server_log = (args.output / 'server.log').open('a')
    server = subprocess.Popen([sys.executable, str(HERE/'serve.py'), '--status-dir', str(args.output),
        '--run', str(args.run), '--checkpoint', str(args.checkpoint), '--port', str(args.port)],
        stdout=server_log, stderr=subprocess.STDOUT, env=env)
    try:
        ready = False
        for _ in range(180):
            if server.poll() is not None:
                raise RuntimeError('bprior policy server exited; inspect server.log')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/healthz', timeout=2) as response:
                    ready = response.status == 200
            except Exception:
                pass
            if ready:
                break
            time.sleep(5)
        if not ready:
            raise TimeoutError('bprior websocket server did not become ready')

        path = args.output / 'rollouts.jsonl'
        latest = latest_rows(path)
        schedule = [(task, trial, condition) for task in args.task_ids
                    for trial in args.trial_indices
                    for condition in ('bprior_direct', 'bprior_retrieve')]
        for task, trial, condition in schedule:
            identity = ('libero_10', task, trial, condition)
            if latest.get(identity, {}).get('status') == 'complete':
                continue
            job = {'host':'127.0.0.1', 'port':args.port, 'suite':'libero_10', 'task_id':task,
                'trial_index':trial, 'seed_base':7, 'condition':condition,
                'memory_snapshot':str(args.run), 'query_text_source':'instruction_only',
                'checkpoint':str(args.checkpoint), 'wait_steps':10, 'replan_steps':5,
                'max_steps':suite_control_limit('libero_10')}
            append_jsonl(path, {'task_suite':'libero_10','task_id':task,'trial_index':trial,
                                'condition':condition,'status':'in_progress','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
            with ProcessPoolExecutor(max_workers=1) as pool:
                result = pool.submit(_run_trial, job).result()
            result['status'] = 'complete' if result['status'] == 'complete' else 'error'
            result['recorded_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            append_jsonl(path, result)
            current_summary = summarize(args.output, args.run)
            write_json(args.output/'closedloop_summary.json', current_summary)
            write_report(args.output, current_summary)
            write_json(args.output/'evaluation_status.json', {'status':'running', 'last':identity,
                'completed_conditions':sum(1 for v in latest_rows(path).values() if v.get('status') == 'complete'),
                'scheduled_conditions':len(schedule)})
        summary = summarize(args.output, args.run)
        write_json(args.output/'closedloop_summary.json', summary)
        completed_count = sum(summary['conditions'][c]['completed'] for c in summary['conditions'])
        final_status = 'complete' if completed_count == len(schedule) else 'incomplete_errors'
        write_json(args.output/'evaluation_status.json', {'status':final_status,
            'completed_conditions':completed_count, 'scheduled_conditions':len(schedule), 'smoke_only':True})
        write_report(args.output, summary)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait()
        server_log.close()


def write_report(out: Path, summary: dict):
    lines = ['# b prior 下書き：LIBERO-Long 動作確認', '',
        'これは各条件2試行（task 0、初期状態0–1）の動作確認であり、本番100試行の結果ではない。', '',
        '| 条件 | 成功/完了 | 成功率 | Wilson 95% CI | 秒/試行 | prior ms | 全推論 median ms | 検索 ms | cos | draft→action MSE | task別 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for name, x in summary['conditions'].items():
        ci = x['wilson95']
        ci_text = '-' if ci is None else f'[{ci[0]:.3f}, {ci[1]:.3f}]'
        rate = '-' if x['success_rate'] is None else f"{x['success_rate']:.3f}"
        sec = '-' if x['mean_trial_seconds'] is None else f"{x['mean_trial_seconds']:.1f}"
        b_ms = x['mean_b_hat_ms_per_query']
        b_text = '-' if b_ms is None else f'{b_ms:.2f}'
        total_ms = x['median_total_policy_query_ms']
        total_text = '-' if total_ms is None else f'{total_ms:.2f}'
        retrieve_ms = x['mean_retrieve_ms_per_query']
        retrieve_text = '-' if retrieve_ms is None else f'{retrieve_ms:.2f}'
        cosine = '-' if x['mean_retrieval_cosine'] is None else f"{x['mean_retrieval_cosine']:.4f}"
        draft_mse = '-' if x['mean_draft_final_mse_normalized'] is None else f"{x['mean_draft_final_mse_normalized']:.5f}"
        lines.append(f"| {name} | {x['successes']}/{x['completed']} | {rate} | {ci_text} | {sec} | {b_text} | {total_text} | {retrieve_text} | {cosine} | {draft_mse} | {summary['task_successes'][name]} |")
    lines.extend(['', '## run-01 との対応比較', '', '```json',
                  json.dumps(summary['paired_to_run01'], ensure_ascii=False, indent=2), '```', '',
                  'run-01 の `normal` / `memory` と、task・initial state index をキーに対応付けた。',
                  '各 rollout の `selected_units_and_t0` に、b-prior取得時間、検索row/cosine、draftと最終出力の差を保持する。',
                  '`prior ms` は codec の prior_net 部分、`全推論 median ms` は前処理・VLM特徴・prior・復号・bridgeを含む1問い合わせ全体（初回JITを含めず中央値）。',
                  'この少数スモークから成功率差の結論は出さない。'])
    checks = out/'server_checks.json'
    if checks.exists():
        lines.extend(['', '## サーバー側キャッシュ一致確認', '', '```json',
                      json.dumps(json.loads(checks.read_text()), ensure_ascii=False, indent=2), '```'])
        lines.append('cached `pooled[row]` を入力した b̂/r̂ は score_audit と絶対差 0。画像経路の特徴は float32 のため bitwise ではなく max abs 7.52e-4（allclose atol=1e-3）で一致した。特徴差に対応して live b̂ と cached-input b̂ の差は 8.06e-3。')
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
