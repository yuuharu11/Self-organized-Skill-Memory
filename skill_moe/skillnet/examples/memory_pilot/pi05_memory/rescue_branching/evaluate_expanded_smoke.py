"""One-failure smoke for the expanded 44-condition rescue branch matrix."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PI05))
sys.path.insert(0, str(PI05.parent/'anchor_flow'))
from key_merge import KEY_ORDER, raw_memory_draft
from run import load_rows
from rollout import build_env, observation_sha, request
from run_case import S2WebsocketClientPolicy, episode_status

ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/rescue-branching-01')
RUN = ROOT.parent/'run-01'
PAIR = ROOT.parent/'paired-breakdown-01'/'normal_failures.jsonl'
CHECKPOINT = Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')
# resume-check-08 is the verified deterministic-replay snapshot. The root
# trials/ checkpoint is retained as a first-attempt diagnostic only.
TRIAL_ROOT = ROOT/'resume-check-08'/'trials'/'task06_init00'


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def env_for(threads='8'):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
        XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='0.15',
        BRANCH_TORCH_MEM_FRACTION='0.20')
    return env


def wait_ready(proc, port, log_path):
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f'expanded policy server exited; inspect {log_path}')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f'expanded server did not start; inspect {log_path}')


def paired_failure():
    for line in PAIR.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if (item.get('task_suite') == 'libero_10' and int(item.get('task_id', -1)) == 6
                and int(item.get('initial_state_index', -1)) == 0):
            if item.get('normal_failure', {}).get('termination_reason') != 'control_budget':
                raise ValueError('selected source is not a normal-policy control-budget failure')
            return item
    raise FileNotFoundError('no paired-breakdown normal failure for task 6 / initial 0')


def rank_at_saved_checkpoint(port, cp_path, replay_path, out_dir):
    replay = json.loads(Path(replay_path).read_text())
    cp = np.load(cp_path, allow_pickle=False)
    task, trial = int(cp['task_id']), int(cp['trial_index'])
    seed = int(cp['trial_seed'])
    env, obs, language, _, _, _, done = build_env(task, trial, seed-trial)
    client = None
    try:
        steps, query_id = 0, int(cp['query_id'])
        for query in replay['queries']:
            if int(query['query_id']) >= query_id:
                break
            for action in np.asarray(query['actions'], np.float32)[:5]:
                obs, _, rd, _ = env.step(action.tolist())
                steps += 1
                _, done = episode_status(env, rd)
                if done:
                    break
            if done:
                break
        if steps != int(cp['env_steps']):
            raise AssertionError(f'replayed {steps} frames, expected {int(cp["env_steps"])}')
        if observation_sha(obs) != str(cp['observation_sha256']):
            raise AssertionError('observation does not match saved branch checkpoint')
        flat = np.asarray(env.sim.get_state().flatten(), np.float64)
        flat_delta = float(np.max(np.abs(flat-np.asarray(cp['sim_state'], np.float64))))
        if flat_delta > 1e-12:
            raise AssertionError(f'simulator checkpoint mismatch {flat_delta}')
        client = S2WebsocketClientPolicy('127.0.0.1', port)
        request_value = request(obs, language, task, trial, query_id, 'normal_rank', seed_base=seed-trial)
        response = client.infer(request_value)
        generated = np.asarray(response['actions'], np.float32)[:, :7]
        original_query = next(q for q in replay['queries'] if int(q['query_id']) == query_id)
        original = np.asarray(original_query['actions'], np.float32)[:, :7]
        if not np.array_equal(generated, original):
            raise AssertionError('expanded server changed normal policy action at checkpoint query')
        debug = response['s2_debug']
        candidates = debug['merged_candidates']
        rows = load_rows(RUN)
        checks = []
        for candidate in candidates:
            row_id = int(candidate['memory_row'])
            raw = raw_memory_draft(rows['actions'], row_id)
            # This is exactly memory_gap/run.py's Draw_* construction:
            # np.asarray(rows['actions'][candidate], np.float32).
            expected = np.asarray(rows['actions'][row_id], np.float32)
            if not np.array_equal(raw, expected):
                raise AssertionError(f'D-raw differs from memory_gap rows[action] for row {row_id}')
            checks.append({'memory_row':row_id,'shape':list(raw.shape),
                'equal_to_memory_gap_D_raw':True,
                'draft_sha256':hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
                'key_ranks':candidate['key_ranks']})
        if len(candidates) != 8 or len({x['memory_row'] for x in candidates}) != 8:
            raise AssertionError('candidate union must contain eight distinct memory rows')
        rankings = {'query_id':query_id,'branch_env_steps':steps,
            'checkpoint_observation_sha256':observation_sha(obs),
            'normal_action_matches_saved_query':True,
            'sim_state_max_abs_difference':flat_delta,
            'per_key_top8':debug['four_key_top8'],'merged_candidates':checks,
            'source_definitions':{
                'vlm_current':'grouped token MaxSim, equal mean of base/wrist/instruction (existing current key)',
                'wrist_only':'token MaxSim over wrist group only (score_audit/make_vlm_keys.py)',
                'codec_prior_b':'cosine of memory-standardized codec prior b-hat (score_audit/keys.py generate_codec)',
                'pi05_pred_to_memory_action':'negative MSE, official pi05 predicted active-7 action vs recorded normalized memory active-7 action (score_audit/keys.py generate_policy_keys)'},
            'ranking_scores_are_higher_better':True}
        target = Path(out_dir)/'four_key_candidates.json'
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(rankings, indent=2, ensure_ascii=False)+'\n')
        return rankings
    finally:
        if client is not None:
            client.close()
        env.close()


def run_job(job, *, port, cp_path, replay_path, out_dir, index, timeout=600):
    branch_dir = out_dir/'branches'
    branch_dir.mkdir(parents=True, exist_ok=True)
    path = branch_dir/f'{index:02d}_{job["name"]}.json'
    log = path.with_suffix('.log')
    if path.exists():
        raise FileExistsError(path)
    # The controller uses 8 threads, the server 4, and each of four workers 5:
    # total allocation is capped at 32 cores.
    command = ['taskset','-c',('12-16','17-21','22-26','27-31')[index%4],
        'nice','-n','10',sys.executable,str(HERE/'rollout_extended.py'),
        '--port',str(port),'--checkpoint',str(cp_path),'--replay',str(replay_path),
        '--condition',job['condition'],'--output',str(path)]
    if job.get('memory_row') is not None:
        command += ['--memory-row',str(job['memory_row']),'--t0',str(job['t0']),
                    '--candidate-sources-json',str(job['sources_path'])]
    elif job.get('t0') is not None:
        command += ['--t0',str(job['t0'])]
    if job.get('seed_offset'):
        command += ['--seed-offset',str(job['seed_offset'])]
    started = time.monotonic()
    with log.open('w') as handle:
        try:
            proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                env=env_for(threads='5'), timeout=timeout, start_new_session=True)
        except subprocess.TimeoutExpired:
            return {'name':job['name'],'status':'timeout','elapsed_seconds':float(timeout),'log':str(log)}
    if proc.returncode != 0 or not path.exists():
        return {'name':job['name'],'status':'error','returncode':proc.returncode,
                'elapsed_seconds':time.monotonic()-started,'log':str(log)}
    result = json.loads(path.read_text())
    result.update(name=job['name'],log=str(log),result_path=str(path))
    return result


def make_jobs(candidate_meta, source_path):
    jobs = [{'name':f'normal_alt_rng_{i+1:02d}','condition':'normal_alt','seed_offset':1009*(i+1)}
            for i in range(8)]
    for t0 in (.6,.2):
        jobs.append({'name':f'zero_t0_{t0:.1f}','condition':'zero_fixed','t0':t0})
        jobs.append({'name':f'current_vlm_t0_{t0:.1f}','condition':'current_fixed','t0':t0})
    for ix, candidate in enumerate(candidate_meta):
        row = int(candidate['memory_row'])
        source_meta = candidate['key_ranks']
        provenance_dir = source_path.parent/'candidate_provenance'
        provenance_dir.mkdir(parents=True, exist_ok=True)
        provenance_path = provenance_dir/f'row_{row}.json'
        provenance_path.write_text(json.dumps({'memory_row':row,'key_ranks':source_meta},
                                               indent=2,ensure_ascii=False)+'\n')
        for fmt, condition in [('codec','candidate_codec'),('raw','candidate_raw')]:
            for t0 in (.4,.2):
                jobs.append({'name':f'candidate{ix+1:02d}_row{row}_{fmt}_t0_{t0:.1f}',
                    'condition':condition,'memory_row':row,'t0':t0,'sources_path':provenance_path,
                    'candidate_sources':source_meta})
    assert len(jobs) == 44
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'expanded-smoke-01')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--port', type=int)
    a = p.parse_args()
    if not 1 <= a.workers <= 4:
        raise ValueError('workers must be 1..4')
    out_dir = a.output
    if out_dir.exists():
        raise FileExistsError(f'refusing to overwrite smoke output {out_dir}')
    out_dir.mkdir(parents=True)
    source = paired_failure()
    replay_path = TRIAL_ROOT/'normal_replay.json'
    cp_path = TRIAL_ROOT/'branchpoints'/'at_60.npz'
    if not replay_path.exists() or not cp_path.exists():
        raise FileNotFoundError('previously verified 60% replay/checkpoint is missing')
    replay = json.loads(replay_path.read_text())
    if replay['success'] or replay['initial_state_sha256'] != source['initial_state_sha256']:
        raise AssertionError('saved normal replay no longer matches selected failure source')
    port = a.port or free_port()
    server_log = out_dir/'server.log'
    with server_log.open('w') as handle:
        server = subprocess.Popen(['taskset','-c','8-11',sys.executable,str(HERE/'serve_extended.py'),
            '--run',str(RUN),'--checkpoint',str(CHECKPOINT),'--port',str(port)],
            stdout=handle,stderr=subprocess.STDOUT,env=env_for(threads='4'),start_new_session=True)
    try:
        wait_ready(server, port, server_log)
        ranking = rank_at_saved_checkpoint(port, cp_path, replay_path, out_dir)
        source_path = out_dir/'four_key_candidates.json'
        jobs = make_jobs(ranking['merged_candidates'], source_path)
        # Preserve the previously required same-RNG resume check as a separate
        # extra branch, not counted among the 44 requested conditions.
        same_rng = {'name':'normal_same_rng_resume','condition':'normal'}
        all_jobs = jobs+[same_rng]
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(run_job, job, port=port, cp_path=cp_path,
                replay_path=replay_path, out_dir=out_dir, index=i) for i,job in enumerate(all_jobs)]
            results = [f.result() for f in as_completed(futures)]
        results.sort(key=lambda x:x['name'])
        completed = [x for x in results if x.get('status') == 'complete']
        required_results = [x for x in results if x['name'] != same_rng['name']]
        if len(required_results) != 44 or any(x.get('status') != 'complete' for x in results):
            raise RuntimeError('expanded branch smoke did not complete all requested conditions')
        original_query = int(np.load(cp_path, allow_pickle=False)['query_id'])
        baseline_hashes = {int(x['query_id']):x['sha256'] for x in replay['queries']
                           if int(x['query_id']) >= original_query}
        same = next(x for x in results if x['name'] == same_rng['name'])
        resumed_hashes = {int(x['query_id']):x['action_sha256'] for x in same.get('query_chunks',[])}
        matches = sum(baseline_hashes.get(q) == value for q,value in resumed_hashes.items())
        if not resumed_hashes or matches != len(resumed_hashes):
            raise AssertionError(f'same-RNG branch differs: {matches}/{len(resumed_hashes)} chunks')
        elapsed = time.monotonic()-started
        branch_times = [float(x['elapsed_seconds']) for x in required_results]
        mean_s = float(np.mean(branch_times))
        max_s = float(np.max(branch_times))
        total_production = 10*3*44
        estimate_s = total_production*mean_s/a.workers + 10*float(replay['elapsed_seconds'])
        result = {'status':'complete','source':{
            'task_id':int(source['task_id']),'initial_state_index':int(source['initial_state_index']),
            'trial_seed':int(source['trial_seed']),'initial_state_sha256':source['initial_state_sha256'],
            'original_normal_failure_steps':int(replay['env_steps'])},
            'branch_point':{'fraction':.6,'env_steps':int(np.load(cp_path,allow_pickle=False)['env_steps']),
                            'query_id':original_query,'checkpoint':str(cp_path)},
            'condition_counts':{'normal_alt_rng':8,'zero_fixed_t0_0.6_0.2':2,
                'current_vlm_fixed_t0_0.6_0.2':2,'memory_candidates':8,
                'candidate_forms':['codec_decode','raw_rows_actions'],'candidate_t0':[.4,.2],
                'requested_conditions':44,'extra_same_rng_resume':1},
            'same_rng_resume':{'action_chunks_matching':matches,
                'action_chunks_total':len(resumed_hashes),'all_match':True},
            'completed':len(completed),'errors':sum(x.get('status')=='error' for x in results),
            'timeouts':sum(x.get('status')=='timeout' for x in results),
            'branch_mean_seconds':mean_s,'branch_max_seconds':max_s,
            'smoke_wall_seconds':elapsed,'workers':a.workers,
            'production_estimate':{'max_trials':10,'branch_points_per_trial':3,
                'conditions_per_point':44,'total_branches':total_production,
                'parallel_workers':a.workers,'branch_wall_hours':total_production*mean_s/a.workers/3600,
                'plus_10_failure_captures_hours':10*float(replay['elapsed_seconds'])/3600,
                'estimated_total_hours':estimate_s/3600},
            'candidate_manifest':str(source_path),'server_log':str(server_log),
            'results':[{'name':x['name'],'status':x.get('status'),'success':x.get('success'),
                'steps':x.get('env_steps'),'memory_row':x.get('memory_row'),'t0':x.get('t0'),
                'elapsed_seconds':x.get('elapsed_seconds'),'result_path':x.get('result_path'),
                'log':x.get('log')} for x in results]}
        result_path = out_dir/'smoke_results.json'
        result_path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        report = [
            '# Expanded rescue-branching smoke', '',
            '単一のLIBERO-Simulation失敗試行・60%分岐点での実装確認。閉ループ本番ではない。', '',
            '| 確認 | 結果 |','|---|---|',
            f"| 分岐元 | task {source['task_id']} init {source['initial_state_index']}, seed {source['trial_seed']}, normal失敗 {replay['env_steps']} step |",
            f"| 分岐点 | step {result['branch_point']['env_steps']} / query {original_query} |",
            f"| 条件 | 44/44完走、errors={result['errors']}、timeouts={result['timeouts']} |",
            f"| 同一乱数の再開 | {matches}/{len(resumed_hashes)} action chunks一致 |",
            f"| 1分岐の時間 | 平均 {mean_s:.1f}s、最大 {max_s:.1f}s |",
            f"| 最大規模の概算 | 10試行×3点×44条件={total_production}枝、4並列で約 {estimate_s/3600:.2f}h（失敗試行の再生を加算） |",
            '| D-raw一致 | 8候補すべてで `rows[\'actions\'][memory_row]` とshape `[10,32]` が完全一致 |', '',
            '4キーは現行VLM grouped MaxSim、手首token MaxSim、memory統計で標準化したcodec prior b̂ cosine、公式π0.5予測active-7と記録行動active-7の負MSE。各候補の全キー順位・scoreを `four_key_candidates.json` に保存。',
            '一試行だけの動作確認であり、成功率の推定には使わない。既存の互換経路を保持し、今回の枝は新しい `serve_extended.py` / `rollout_extended.py` 経由。', '',
            f'詳細: `{result_path}`。候補一覧: `{source_path}`。']
        (out_dir/'report.md').write_text('\n'.join(report)+'\n')
        print(json.dumps({'output':str(out_dir),'completed':len(completed),
            'errors':result['errors'],'timeouts':result['timeouts'],
            'mean_branch_seconds':mean_s,'estimated_production_hours':estimate_s/3600},ensure_ascii=False))
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            server.wait()


if __name__ == '__main__':
    main()
