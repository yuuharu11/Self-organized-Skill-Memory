"""Resumable two-server adaptive-noise closed-loop smoke/full runner.

The default CLI schedules all 10 x 10 rollouts per condition. Exp owns the
full run; use --smoke to execute two unique trials per condition and repeat
them concurrently to check that request ordering does not alter actions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
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
ROOT = _PI05_ROOT
RUN = ROOT / 'run-01'
OUTPUT = ROOT / 'adaptive-long-01'
CHECKPOINT = _STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero'
CONDITIONS = ('adaptive', 'adaptive_fb')


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def append_jsonl(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
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
            latest[(row['task_suite'], int(row['task_id']), int(row['trial_index']), row['condition'])] = row
    return latest


def exact_mcnemar(candidate, baseline):
    c_only = sum(c == 1 and b == 0 for c, b in zip(candidate, baseline))
    b_only = sum(c == 0 and b == 1 for c, b in zip(candidate, baseline))
    discordant = c_only + b_only
    p = (min(1.0, 2 * sum(math.comb(discordant, i)
                          for i in range(min(c_only, b_only) + 1)) / 2**discordant)
         if discordant else 1.0)
    return {'candidate_only_success': c_only, 'baseline_only_success': b_only,
            'both_success': sum(c == b == 1 for c, b in zip(candidate, baseline)),
            'both_failure': sum(c == b == 0 for c, b in zip(candidate, baseline)),
            'paired_n': len(candidate), 'exact_mcnemar_p': p}


def wilson(successes, n):
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [center-half, center+half]


def summarize(out: Path, run: Path, bprior: Path):
    current = latest_rows(out / 'rollouts.jsonl')
    history = ([json.loads(line) for line in (out/'rollouts.jsonl').read_text().splitlines()]
               if (out/'rollouts.jsonl').exists() else [])
    baselines = {'run01_normal': latest_rows(run / 'rollouts.jsonl'),
                 'run01_memory': latest_rows(run / 'rollouts.jsonl'),
                 'bprior_direct': latest_rows(bprior / 'rollouts.jsonl'),
                 'bprior_retrieve': latest_rows(bprior / 'rollouts.jsonl')}
    key_for = {
        'run01_normal': lambda t, i: ('libero_10', t, i, 'normal'),
        'run01_memory': lambda t, i: ('libero_10', t, i, 'memory'),
        'bprior_direct': lambda t, i: ('libero_10', t, i, 'bprior_direct'),
        'bprior_retrieve': lambda t, i: ('libero_10', t, i, 'bprior_retrieve')}
    summary = {'conditions': {}, 'task_successes': {}, 'paired': {},
               'latency_seconds': {}, 'noise_usage': {}, 'error_counts': {}}
    for cond in CONDITIONS:
        records = [r for k, r in current.items() if k[3] == cond]
        completed = [r for r in records if r.get('status') == 'complete']
        success = sum(bool(r.get('success')) for r in completed)
        summary['conditions'][cond] = {'successes': success, 'completed': len(completed),
            'success_rate': success/len(completed) if completed else None,
            'wilson95': wilson(success, len(completed)), 'scheduled': len(records),
            'errors': sum(r.get('status') in ('error', 'timeout') for r in records)}
        summary['error_counts'][cond] = {
            'error_attempts': sum(r.get('condition') == cond and r.get('status') == 'error'
                                  for r in history),
            'timeout_attempts': sum(r.get('condition') == cond and r.get('status') == 'timeout'
                                    for r in history)}
        by_task = {}
        for task in sorted({int(r['task_id']) for r in completed}):
            group = [r for r in completed if int(r['task_id']) == task]
            by_task[str(task)] = {'successes': sum(bool(r['success']) for r in group), 'n': len(group)}
        summary['task_successes'][cond] = by_task
        traces = [x for r in completed for x in r.get('selected_units_and_t0', [])
                  if x.get('condition') == cond]
        mode_counts = {'fallback': 0, 't0_0.6': 0, 't0_0.4': 0, 't0_0.2': 0}
        for tr in traces:
            if tr.get('fallback'):
                mode_counts['fallback'] += 1
            elif tr.get('t0') is not None:
                mode_counts[f"t0_{float(tr['t0']):.1f}"] += 1
        ntr = len(traces)
        summary['noise_usage'][cond] = {'queries': ntr, 'counts': mode_counts,
            'rates': {k: (v/ntr if ntr else None) for k, v in mode_counts.items()},
            'fallback_rate': mode_counts['fallback']/ntr if ntr else None}
        components = ('preprocess_seconds', 'vlm_seconds', 'search_seconds', 'draft_seconds',
                      'denoise_seconds', 'output_transform_seconds', 'total_seconds')
        lat = {}
        for name in components:
            vals = [float(t[name]) for t in traces if name in t]
            lat[name] = {'mean_ms': 1000*sum(vals)/len(vals) if vals else None,
                         'median_ms': 1000*sorted(vals)[len(vals)//2] if vals else None}
        elapsed = [float(r.get('elapsed_seconds', 0)) for r in completed]
        rpc = [float(r.get('websocket_roundtrip_seconds', 0)) for r in completed]
        env_only = [float(r.get('environment_reset_and_step_seconds', 0)) for r in completed]
        policy = [sum(float(t.get('total_seconds', 0)) for t in r.get('selected_units_and_t0', [])
                      if t.get('condition') == cond) for r in completed]
        residual = [max(0., a-b) for a, b in zip(elapsed, policy)]
        lat['environment_and_websocket_residual'] = {
            'mean_seconds_per_trial': sum(residual)/len(residual) if residual else None,
            'note': 'trial wall time minus server inference; includes simulator, image transfer and websocket overhead'}
        lat['websocket_roundtrip'] = {'mean_seconds_per_trial': sum(rpc)/len(rpc) if rpc else None}
        lat['environment_reset_and_step'] = {
            'mean_seconds_per_trial': sum(env_only)/len(env_only) if env_only else None}
        summary['latency_seconds'][cond] = lat
        for baseline_name, base in baselines.items():
            pairs = []
            for r in completed:
                key = key_for[baseline_name](int(r['task_id']), int(r['trial_index']))
                b = base.get(key)
                if b and b.get('status') == 'complete':
                    if b.get('initial_state_sha256') != r.get('initial_state_sha256'):
                        raise ValueError(f'initial state mismatch: {baseline_name} vs {cond}')
                    if b.get('trial_seed') != r.get('trial_seed'):
                        raise ValueError(f'trial seed mismatch: {baseline_name} vs {cond}')
                    pairs.append((r, b))
            summary['paired'][f'{cond}_vs_{baseline_name}'] = exact_mcnemar(
                [int(r['success']) for r, _ in pairs], [int(b['success']) for _, b in pairs])
    return summary


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def wait_ready(proc, port, log_path):
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f'adaptive server exited (log {log_path})')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f'adaptive server not ready on port {port}')


def run_trial(job, out, attempt, timeout=600):
    key = f"task{job['task_id']:02d}-init{job['trial_index']:02d}-{job['condition']}"
    folder = out / 'attempts' / key / f'attempt-{attempt}'
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/'job.json', job)
    env = os.environ.copy()
    threads = '2'
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = env['OPENBLAS_NUM_THREADS'] = threads
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    worker_log = (folder/'worker.log').open('w')
    started = time.time()
    process = subprocess.Popen(['taskset', '-c', '20-35', sys.executable, str(HERE/'trial_worker.py'),
        str(folder/'job.json'), str(folder/'result.json')], stdout=worker_log,
        stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        process.wait(timeout=timeout)
        if process.returncode != 0 or not (folder/'result.json').exists():
            return {'status': 'error', 'error': f'worker exit code {process.returncode}',
                    'elapsed_seconds': time.time()-started, 'attempt': attempt,
                    'job': job, 'attempt_dir': str(folder)}
        result = json.loads((folder/'result.json').read_text())
        result.update(attempt=attempt, attempt_dir=str(folder))
        return result
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        return {'status': 'timeout', 'error': f'rollout exceeded {timeout}s',
                'elapsed_seconds': time.time()-started, 'attempt': attempt,
                'job': job, 'attempt_dir': str(folder)}
    finally:
        worker_log.close()


def server_env():
    env = os.environ.copy()
    # 8 environment workers x 3 threads + 2 policy servers x 2 threads +
    # controller's 3 BLAS threads = 31 runnable CPU threads at peak.
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = env['OPENBLAS_NUM_THREADS'] = '2'
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # JAX's BFC pool and the torch-resident token index share a per-server cap.
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.12'
    env['ADAPTIVE_TORCH_MEM_FRACTION'] = '0.20'
    return env


def append_record(out, result, status='complete', job=None):
    if job is not None:
        result.setdefault('task_suite', 'libero_10')
        result.setdefault('task_id', int(job['task_id']))
        result.setdefault('trial_index', int(job['trial_index']))
        result.setdefault('condition', job['condition'])
        result.setdefault('trial_seed', int(job['seed_base']) + int(job['trial_index']))
        result.setdefault('success', False)
    result.setdefault('recorded_at_utc', now())
    result['status'] = status
    append_jsonl(out/'rollouts.jsonl', result)


def run_jobs(out, run, bprior, jobs, ports, max_workers=8, timeout=600, consistency=False):
    logs = {}
    servers = {}
    try:
        server_cpus = {'adaptive': '8-13', 'adaptive_fb': '14-19'}
        for condition in CONDITIONS:
            status_dir = out/'servers'/condition
            status_dir.mkdir(parents=True, exist_ok=True)
            port = ports[condition]
            log = (out/'servers'/f'{condition}.log').open('a')
            logs[condition] = log
            proc = subprocess.Popen(['taskset', '-c', server_cpus[condition], sys.executable,
                str(HERE/'serve.py'), '--run', str(run),
                '--checkpoint', str(CHECKPOINT), '--status-dir', str(status_dir),
                '--condition', condition, '--port', str(port)], stdout=log,
                stderr=subprocess.STDOUT, env=server_env(), start_new_session=True)
            servers[condition] = proc
            wait_ready(proc, port, out/'servers'/f'{condition}.log')

        if consistency:
            # Serial baseline, then the exact same task/initial pairs together.
            primary = jobs
            for job in primary:
                append_jsonl(out/'rollouts.jsonl', {'task_suite':'libero_10',
                    'task_id':job['task_id'],'trial_index':job['trial_index'],
                    'condition':job['condition'],'status':'in_progress','attempt':1,
                    'phase':'serial','utc':now()})
                result = run_trial(job, out, 1, timeout)
                append_record(out, result, result.get('status','error'), job)
            parallel_jobs = [(dict(j, consistency_phase='parallel'), replica)
                             for j in primary for replica in (0, 1)]
            # Four independent simulators per server. Each key is repeated only
            # for determinism verification and is not counted as another trial.
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(run_trial, job, out, 2 + replica, timeout):
                           (job, replica) for job, replica in parallel_jobs}
                parallel_results = [(f.result(), futures[f]) for f in as_completed(futures)]
            for result, (job, replica) in parallel_results:
                result['consistency_phase'] = 'parallel'
                result['parallel_replica'] = replica
                append_record(out, result, result.get('status','error'), job)
            consistency_report(out)
        else:
            latest = latest_rows(out/'rollouts.jsonl')
            scheduled = len(jobs)
            completed = sum(1 for j in jobs if latest.get(('libero_10', j['task_id'],
                j['trial_index'], j['condition']), {}).get('status') == 'complete')
            write_json(out/'evaluation_status.json', {'status':'running','completed':completed,
                'scheduled':scheduled,'started_utc':now()})
            todo = []
            for job in jobs:
                key = ('libero_10', job['task_id'], job['trial_index'], job['condition'])
                if latest.get(key, {}).get('status') == 'complete':
                    continue
                prev = latest.get(key, {})
                attempt = int(prev.get('attempt', 0)) + 1
                if attempt > 2:
                    continue
                todo.append((job, attempt))
            for j, attempt in todo:
                append_jsonl(out/'rollouts.jsonl', {'task_suite':'libero_10',
                    'task_id':j['task_id'],'trial_index':j['trial_index'],'condition':j['condition'],
                    'status':'in_progress','attempt':attempt,'utc':now()})
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for job, attempt in todo:
                    futures[pool.submit(run_trial, job, out, attempt, timeout)] = (job, attempt)
                for future in as_completed(futures):
                    job, attempt = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {'status':'error','error':repr(exc),'job':job,'attempt':attempt}
                    key = ('libero_10',job['task_id'],job['trial_index'],job['condition'])
                    append_record(out, result, result.get('status','error'), job)
                    if result.get('status') != 'complete' and attempt < 2:
                        time.sleep(10)
                        retry = run_trial(job, out, attempt+1, timeout)
                        append_record(out, retry, retry.get('status','error'), job)
                    current = latest_rows(out/'rollouts.jsonl')
                    done = sum(1 for j in jobs if current.get(('libero_10', j['task_id'],
                        j['trial_index'], j['condition']), {}).get('status') == 'complete')
                    write_json(out/'evaluation_status.json', {'status':'running','completed':done,
                        'scheduled':scheduled,'last':[job['task_id'],job['trial_index'],job['condition']],
                        'updated_utc':now()})
                    summary = summarize(out, run, bprior)
                    write_json(out/'closedloop_summary.json', summary)
                    write_report(out, summary, '評価中')
            current = latest_rows(out/'rollouts.jsonl')
            done = sum(1 for j in jobs if current.get(('libero_10', j['task_id'],
                j['trial_index'], j['condition']), {}).get('status') == 'complete')
            status = 'complete' if done == scheduled else 'incomplete_errors'
            write_json(out/'evaluation_status.json', {'status':status,'completed':done,
                'scheduled':scheduled,'updated_utc':now()})
            summary = summarize(out, run, bprior)
            write_json(out/'closedloop_summary.json', summary)
            write_report(out, summary, '完了' if status == 'complete' else status)
    finally:
        for proc in servers.values():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
        for log in logs.values():
            log.close()


def consistency_report(out):
    grouped = {}
    for path in sorted((out/'servers').glob('*/action_trace.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row['condition'],row['episode_id'],int(row['query_id']))
            grouped.setdefault(key, []).append(row['action_sha256'])
    records = []
    for key, hashes in sorted(grouped.items()):
        if len(hashes) >= 2:
            replica_equal = [h == hashes[0] for h in hashes[1:]]
            records.append({'condition':key[0],'episode_id':key[1],'query_id':key[2],
                            'serial_sha256':hashes[0],'parallel_sha256s':hashes[1:],
                            'parallel_replicas_equal':replica_equal,
                            'equal':all(replica_equal)})
    result = {'compared_query_chunks':len(records),
              'matching_query_chunks':sum(r['equal'] for r in records),
              'all_equal':bool(records) and all(r['equal'] for r in records),
              'first_mismatch':next((r for r in records if not r['equal']),None),
              'comparisons':records}
    write_json(out/'parallel_consistency.json', result)
    return result


def write_report(out, summary, status):
    consistency_path = out/'parallel_consistency.json'
    consistency = json.loads(consistency_path.read_text()) if consistency_path.exists() else {}
    consistency_short = {k: consistency.get(k) for k in
                         ('compared_query_chunks','matching_query_chunks','all_equal','first_mismatch')}
    lines = ['# 信頼度スケジュールとフォールバック：LIBERO-Long', '',
        f'状態: {status}。事前オフライン評価では t0=0.2 はほぼ全スコア帯で0.6より悪く、スコアと下書き誤差の相関も想定と逆向きだった。これは仮説の閉ループ検証で、改善を期待した設定ではない。',
        '検索スコアは正解確率ではない。スコア境界は既存 run-01 memory 閉ループの事前固定分位点であり、成功率を使っていない。', '',
        '| 条件 | 成功/完了 | 成功率 | Wilson 95% | タスク別成功 | t0=0.6 / 0.4 / 0.2 | fallback率 |',
        '|---|---:|---:|---:|---|---:|---:|']
    for cond in CONDITIONS:
        x = summary['conditions'][cond]
        ci = x['wilson95']
        usage = summary['noise_usage'][cond]
        rates = usage['rates']
        lines.append(f"| {cond} | {x['successes']}/{x['completed']} | "
            f"{x['success_rate'] if x['success_rate'] is not None else '-'} | "
            f"{ci if ci is not None else '-'} | {summary['task_successes'][cond]} | "
            f"{rates['t0_0.6']} / {rates['t0_0.4']} / {rates['t0_0.2']} | {usage['fallback_rate']} |")
    lines += ['', '## 固定スケジュール', '',
        '| スコア s | adaptive | adaptive_fb |', '|---|---:|---:|',
        '| s < 0.92968 | 0.6 | 通常 π0.5 |',
        '| 0.92968 ≤ s < 0.94267 | 0.6 | 0.6 |',
        '| 0.94267 ≤ s < 0.95002 | 0.4 | 0.4 |',
        '| 0.95002 ≤ s | 0.2 | 0.2 |', '',
        '各推論のスコア・モード・t0・fallback・選択行・行動ハッシュと時間は `servers/<condition>/action_trace.jsonl` に記録。',
        '## 時間内訳', '',
        '| 条件 | VLM ms/query | 検索 ms/query | デノイズ ms/query | websocket 往復 s/trial | env reset/step s/trial |',
        '|---|---:|---:|---:|---:|---:|']
    for cond in CONDITIONS:
        x = summary['latency_seconds'][cond]
        lines.append(f"| {cond} | {x['vlm_seconds']['mean_ms']:.2f} | "
            f"{x['search_seconds']['mean_ms']:.2f} | {x['denoise_seconds']['mean_ms']:.2f} | "
            f"{x['websocket_roundtrip']['mean_seconds_per_trial']:.2f} | "
            f"{x['environment_reset_and_step']['mean_seconds_per_trial']:.2f} |")
    lines += ['','試行時間からサーバー推論時間を引いた残差は、通信・シミュレータを含む参考値。',
        '## 対応比較', '', '対応キーは task と initial index。同じ initial_state_sha256 と trial_seed を検証した。Exact McNemar:', '',
        '```json', json.dumps(summary['paired'], ensure_ascii=False, indent=2), '```', '',
        '## 並列整合性', '', '```json',
        json.dumps(consistency_short, ensure_ascii=False, indent=2),
        '```', '', '少数スモークの成功率は比較の結論に使わない。', '',
        '## 実装・資源メモ', '',
        '既存TokenIndexの全バンクFP32ノルム計算は一時35.76 GiBを要求してTorch allocator上限を超えたため、adaptive側だけをブロック正規化に置き換えた。FP16出力は既存式とbitwise一致するテストを追加。',
        '正常スモーク時のGPU全体ピークは約78.8 GiB（2サーバー合計）。各サーバーはJAX 12%＋Torch 20%のallocator上限。初回失敗起動は推論前の初期化OOMで、ロールアウトには含めない。',
        '最初の起動ログには、run識別子がない出力ディレクトリをcontrollerが拒否した記録も残る。修正後のrun識別子つき `smoke-run2` で評価した。',
        '本番のLIBERO-Long 10×10×2試行はexpがREADMEのtmuxコマンドで実行する。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--bprior', type=Path, default=ROOT/'bprior-long-01'/'production')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--ports', type=int, nargs=2, default=None)
    parser.add_argument('--workers-per-server', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--task-ids', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--trial-indices', type=int, nargs='+', default=list(range(10)))
    args = parser.parse_args()
    if args.smoke:
        args.task_ids = [0]
        args.trial_indices = [0, 1]
    if args.output.exists() and not (args.output/'experiment.json').exists():
        allowed_bootstrap = {'smoke.log', 'smoke.pid'}
        unexpected = {p.name for p in args.output.iterdir()} - allowed_bootstrap
        if unexpected:
            raise FileExistsError(f'output exists but is not this run: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    ports = dict(zip(CONDITIONS, args.ports or [free_port(), free_port()]))
    if ports['adaptive'] == ports['adaptive_fb']:
        raise ValueError('the two policy servers need distinct ports')
    workers = args.workers_per_server * len(CONDITIONS)
    if workers > 8:
        raise ValueError('workers exceed the 32-thread budget (8 workers x 4 threads)')
    protocol = {'suite':'libero_10','conditions':list(CONDITIONS),
        'task_ids':args.task_ids,'trial_indices':args.trial_indices,'seed_base':7,
        'trial_seed':'7 + trial_index','initial_states':'same public initial_states[trial_index] as run-01',
        'replan_steps':5,'wait_steps':10,'max_env_steps':520,'num_steps':10,
        'schedule_edges':[.92968,.94267,.95002],
        'modes':{'adaptive':[1,1,2,3],'adaptive_fb':[0,1,2,3]},
        'smoke':args.smoke,'workers_per_server':args.workers_per_server,
        'rollout_timeout_seconds':args.timeout,'technical_retries':1,
        'checkpoint':str(args.checkpoint),'run_snapshot':str(args.run),
        'GPU_caps':{'XLA_PYTHON_CLIENT_PREALLOCATE':'false',
                    'XLA_PYTHON_CLIENT_MEM_FRACTION':'.12',
                    'ADAPTIVE_TORCH_MEM_FRACTION':'.20',
                    'target_peak_per_server_GB':50,
        'cpu_threads':{'environment_worker':2,'policy_server':2,
                       'server_affinity':{'adaptive':'8-13','adaptive_fb':'14-19'},
                       'worker_affinity':'20-35','controller_threads':3,
                       'max_parallel_workers':workers}},'ports':ports}
    existing = args.output/'experiment.json'
    if existing.exists() and json.loads(existing.read_text()) != protocol:
        raise ValueError('existing experiment metadata differs; refusing to reuse output')
    write_json(existing, protocol)
    # Store immutable source inventory checksums; input run-01 is never written.
    source_files = [args.run/'rollouts.jsonl', args.run/'best_bridge.json', args.run/'calibration.json']
    write_json(args.output/'source_sha256.json', {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_files})
    jobs = []
    if args.smoke:
        for cond in CONDITIONS:
            for trial in args.trial_indices:
                jobs.append({'host':'127.0.0.1','port':ports[cond],'suite':'libero_10',
                    'task_id':0,'trial_index':trial,'seed_base':7,'condition':cond,
                    'memory_snapshot':str(args.run),'query_text_source':'instruction_only',
                    'checkpoint':str(args.checkpoint),'wait_steps':10,'replan_steps':5,
                    'max_steps':520})
        run_jobs(args.output,args.run,args.bprior,jobs,ports,workers, args.timeout, consistency=True)
    else:
        for cond in CONDITIONS:
            for task in args.task_ids:
                for trial in args.trial_indices:
                    jobs.append({'host':'127.0.0.1','port':ports[cond],'suite':'libero_10',
                        'task_id':int(task),'trial_index':int(trial),'seed_base':7,'condition':cond,
                        'memory_snapshot':str(args.run),'query_text_source':'instruction_only',
                        'checkpoint':str(args.checkpoint),'wait_steps':10,'replan_steps':5,
                        'max_steps':520})
        run_jobs(args.output,args.run,args.bprior,jobs,ports,workers,args.timeout,consistency=False)


if __name__ == '__main__':
    main()
