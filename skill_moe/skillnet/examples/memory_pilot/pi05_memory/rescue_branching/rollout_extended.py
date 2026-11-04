"""One counterfactual continuation from a verified replay checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
PI05 = HERE.parent
if str(PI05) not in sys.path:
    sys.path.insert(0, str(PI05))
ANCHOR = HERE.parent.parent / 'anchor_flow'
if str(ANCHOR) not in sys.path:
    sys.path.insert(0, str(ANCHOR))
from key_merge import raw_memory_draft
from rollout import (build_env, episode_status, observation_sha, python_sim_snapshot,
                     request)
from run_case import S2WebsocketClientPolicy


def capture(port, task_id, trial_index, seed_base, output_root, fractions=(.2,.4,.6)):
    """Record an ordinary-policy failure and complete four-key checkpoints."""
    root = Path(output_root)
    folder = root/'trials'/f'task{int(task_id):02d}_init{int(trial_index):02d}'
    if folder.exists():
        raise FileExistsError(f'refusing to reuse/overwrite capture directory {folder}')
    cpdir = folder/'branchpoints'
    cpdir.mkdir(parents=True)
    env, obs, language, initial, _, _, done = build_env(task_id,trial_index,seed_base)
    client = S2WebsocketClientPolicy('127.0.0.1',int(port))
    chunks=[];checkpoints=[];action_plan=[];query_id=0;steps=0;success=False
    targets={int(round(float(frac)*520/5)*5):float(frac) for frac in fractions}
    started=time.monotonic()
    try:
        while steps < 520 and not done:
            if not action_plan:
                is_checkpoint = steps in targets
                condition = 'normal_rank' if is_checkpoint else 'normal'
                req=request(obs,language,task_id,trial_index,query_id,condition,seed_base)
                response=client.infer(req)
                debug=response.get('s2_debug',{})
                if is_checkpoint:
                    if len(debug.get('merged_candidates',[])) != 8:
                        raise AssertionError('expanded server did not return the 8-row merged candidate list')
                    percent=int(round(targets[steps]*100))
                    cp_path=cpdir/f'at_{percent:02d}.npz'
                    flat=np.asarray(env.sim.get_state().flatten(),np.float64)
                    state_sha=hashlib.sha256(flat.tobytes()).hexdigest()
                    np.savez_compressed(cp_path,sim_state=flat,initial_state=initial,
                        env_steps=np.asarray(steps),query_id=np.asarray(query_id),
                        observation_sha256=np.asarray(observation_sha(obs)),
                        observation_agentview=obs['agentview_image'],
                        observation_wrist=obs['robot0_eye_in_hand_image'],
                        observation_state=np.concatenate((obs['robot0_eef_pos'],obs['robot0_eef_quat'],obs['robot0_gripper_qpos'])),
                        python_sim_state=python_sim_snapshot(env),task_id=np.asarray(task_id),
                        trial_index=np.asarray(trial_index),trial_seed=np.asarray(seed_base+trial_index),
                        candidates=np.asarray([x['memory_row'] for x in debug['merged_candidates']],np.int64),
                        scores=np.asarray([x['key_ranks']['vlm_current']['score']
                                           for x in debug['merged_candidates']],np.float32))
                    manifest={'query_id':query_id,'env_steps':steps,'fraction':targets[steps],
                        'four_key_top8':debug['four_key_top8'],
                        'merged_candidates':debug['merged_candidates'],
                        'key_definitions':{
                            'vlm_current':'grouped token MaxSim, equal base/wrist/instruction mean',
                            'wrist_only':'token MaxSim, wrist group only',
                            'codec_prior_b':'cosine after memory-stat standardization of codec prior b-hat',
                            'pi05_pred_to_memory_action':'negative normalized active-7 MSE of official pi0.5 predicted query action to recorded memory action'}}
                    manifest_path=cpdir/f'at_{percent:02d}.keys.json'
                    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
                    checkpoints.append({'fraction':targets[steps],'env_steps':steps,'query_id':query_id,
                        'path':str(cp_path),'manifest_path':str(manifest_path),'state_sha256':state_sha,
                        'candidates':[int(x['memory_row']) for x in debug['merged_candidates']]})
                chunk=np.asarray(response['actions'],np.float32)[:,:7]
                if len(chunk)<5:
                    raise ValueError(f'short action chunk {chunk.shape}')
                action_plan.extend(chunk[:5])
                chunks.append({'query_id':query_id,'seed':(seed_base+trial_index)*100000+query_id,
                    'observation_sha256':observation_sha(obs),
                    'sha256':hashlib.sha256(np.ascontiguousarray(chunk).tobytes()).hexdigest(),
                    'actions':chunk.tolist(),'debug':debug})
                query_id+=1
            obs,_,rd,_=env.step(np.asarray(action_plan.pop(0),np.float32).tolist())
            steps+=1
            success,done=episode_status(env,rd)
        result={'status':'complete','task_id':int(task_id),'trial_index':int(trial_index),
            'trial_seed':int(seed_base+trial_index),'initial_state_sha256':hashlib.sha256(
                np.ascontiguousarray(initial).view(np.uint8)).hexdigest(),
            'success':bool(success),'env_steps':steps,
            'termination_reason':'success' if success else 'timeout_or_done',
            'checkpoints':checkpoints,'queries':chunks,'elapsed_seconds':time.monotonic()-started,
            'task':language}
        if len(checkpoints)!=len(fractions):
            raise RuntimeError(f'normal rollout captured {len(checkpoints)}/{len(fractions)} branch points')
        (folder/'normal_replay.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        return result
    finally:
        client.close();env.close()


def branch(port, checkpoint_path, replay_path, condition, *, output_path,
           memory_row=None, t0=None, candidate_sources=None, seed_offset=0):
    cp_path = Path(checkpoint_path)
    replay = json.loads(Path(replay_path).read_text())
    cp = np.load(cp_path, allow_pickle=False)
    task_id, trial_index = int(cp['task_id']), int(cp['trial_index'])
    trial_seed = int(cp['trial_seed'])
    env, obs, language, _, _, _, done = build_env(task_id, trial_index, trial_seed-trial_index)
    start_steps, query_id = int(cp['env_steps']), int(cp['query_id'])
    client = None
    started = time.monotonic()
    try:
        steps = 0
        for chunk in replay['queries']:
            if int(chunk['query_id']) >= query_id:
                break
            for action in np.asarray(chunk['actions'], np.float32)[:5]:
                obs, _, rd, _ = env.step(action.tolist())
                steps += 1
                _, done = episode_status(env, rd)
                if done:
                    break
            if done:
                break
        if steps != start_steps:
            raise ValueError(f'replay reached step {steps}, checkpoint is {start_steps}')
        flat = np.asarray(env.sim.get_state().flatten(), np.float64)
        delta = float(np.max(np.abs(flat-np.asarray(cp['sim_state'], np.float64))))
        if delta > 1e-12:
            raise ValueError(f'replayed simulator state differs by {delta}')
        if observation_sha(obs) != str(cp['observation_sha256']):
            raise ValueError('replayed observation hash differs from saved checkpoint')
        client = S2WebsocketClientPolicy('127.0.0.1', int(port))
        action_plan, chunks, traces = [], [], []
        success, returned_done = bool(env.check_success()), False
        start_query_id = query_id
        while steps < 520 and not success and not returned_done:
            if not action_plan:
                extras = {}
                if memory_row is not None:
                    extras.update(memory_row=int(memory_row),
                                  candidate_sources=candidate_sources or {})
                if t0 is not None:
                    extras['branch_t0'] = float(t0)
                req = request(obs, language, task_id, trial_index, query_id,
                              condition, seed_base=trial_seed-trial_index, **extras)
                if condition == 'normal_alt':
                    req['s2_rng_seed'] += int(seed_offset)
                response = client.infer(req)
                chunk = np.asarray(response['actions'], np.float32)[:, :7]
                traces.append(response.get('s2_debug', {}))
                chunks.append({'query_id': query_id,
                    'action_sha256': hashlib.sha256(np.ascontiguousarray(chunk).tobytes()).hexdigest()})
                action_plan.extend(chunk[:5])
                query_id += 1
            obs, _, rd, _ = env.step(np.asarray(action_plan.pop(0), np.float32).tolist())
            steps += 1
            success, returned_done = episode_status(env, rd)
        result = {'status':'complete','condition':condition,'memory_row':memory_row,'t0':t0,
            'candidate_sources':candidate_sources or {},'success':bool(success),'env_steps':steps,
            'branch_start_steps':start_steps,'branch_start_query_id':start_query_id,
            'resume_mode':'deterministic_action_replay_to_saved_checkpoint',
            'checkpoint_sim_state_max_abs':delta,'branch_observation_sha256':observation_sha(obs),
            'query_chunks':chunks,'traces':traces,'elapsed_seconds':time.monotonic()-started}
        output_path = Path(output_path)
        if output_path.exists():
            raise FileExistsError(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
        return result
    finally:
        if client is not None:
            client.close()
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--condition', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--memory-row', type=int)
    p.add_argument('--t0', type=float)
    p.add_argument('--candidate-sources-json', type=Path)
    p.add_argument('--seed-offset', type=int, default=0)
    a = p.parse_args()
    sources = json.loads(a.candidate_sources_json.read_text()) if a.candidate_sources_json else {}
    result = branch(a.port, a.checkpoint, a.replay, a.condition, output_path=a.output,
        memory_row=a.memory_row, t0=a.t0, candidate_sources=sources, seed_offset=a.seed_offset)
    print(json.dumps({'condition':result['condition'],'success':result['success'],
                      'steps':result['env_steps'],'seconds':result['elapsed_seconds']}))


if __name__ == '__main__':
    main()
