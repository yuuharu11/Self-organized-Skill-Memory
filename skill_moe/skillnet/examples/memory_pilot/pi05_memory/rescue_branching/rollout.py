"""Capture/replay one normal failure and execute one saved-state branch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
PI05=HERE.parent
ANCHOR=PI05.parent/'anchor_flow'
sys.path.insert(0,str(ANCHOR))
from run_case import (S2WebsocketClientPolicy,_suite_name,_task_language,
    _quat2axisangle,environment_horizon,episode_status)

RUN=Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
CHECKPOINT=Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')
OUT=Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/rescue-branching-01')


def request(obs,language,task_id,trial_index,query_id,condition='normal_rank',seed_base=7,**extra):
    from skillnet_client import image_tools
    image=np.ascontiguousarray(obs['agentview_image'][::-1,::-1])
    wrist=np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1])
    return {'observation/image':image_tools.convert_to_uint8(image_tools.resize_with_pad(image,224,224)),
      'observation/wrist_image':image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist,224,224)),
      'observation/state':np.concatenate((obs['robot0_eef_pos'],_quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos'])),
      'prompt':language,'branch_condition':condition,'s2_episode_id':f'libero_10:{task_id}:{trial_index}',
      's2_query_id':trial_index*1000+query_id,'s2_rng_seed':(seed_base+trial_index)*100000+query_id,**extra}


def observation_sha(obs):
    arrays=[obs['agentview_image'],obs['robot0_eye_in_hand_image'],obs['robot0_eef_pos'],
        obs['robot0_eef_quat'],obs['robot0_gripper_qpos']]
    h=hashlib.sha256()
    for value in arrays:h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def python_sim_snapshot(env):
    """Capture mutable controller/timestep state omitted by MjSimState."""
    inner=env.env
    ctrl=inner.robots[0].controller
    payload={'timestep':int(inner.timestep),'cur_time':float(inner.cur_time),'done':bool(inner.done),
             'controller':{}}
    for key,value in vars(ctrl).items():
        if isinstance(value,np.ndarray):payload['controller'][key]=value.copy()
        elif isinstance(value,(bool,int,float,str,type(None),list,tuple)):
            try:pickle.dumps(value);payload['controller'][key]=value
            except Exception:pass
    try:
        import mujoco
        model=inner.sim.model._model;data=inner.sim.data._data
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        full=np.empty(mujoco.mj_stateSize(model,spec),np.float64)
        mujoco.mj_getState(model,data,full,spec)
        payload['mujoco_integration_state']=full
    except Exception as exc:
        payload['mujoco_integration_state_error']=repr(exc)
    payload['sim_data']={}
    for key in ('qpos','qvel','act','qacc_warmstart','ctrl','qfrc_applied',
                'xfrc_applied','mocap_pos','mocap_quat','userdata','eq_active'):
        if hasattr(inner.sim.data,key):
            value=getattr(inner.sim.data,key)
            if isinstance(value,np.ndarray):payload['sim_data'][key]=value.copy()
    return np.frombuffer(pickle.dumps(payload,protocol=4),dtype=np.uint8).copy()


def restore_python_sim_snapshot(env,blob):
    payload=pickle.loads(np.asarray(blob,dtype=np.uint8).tobytes())
    inner=env.env
    inner.timestep=payload['timestep'];inner.cur_time=payload['cur_time'];inner.done=payload['done']
    full=payload.get('mujoco_integration_state')
    if full is not None:
        import mujoco
        model=inner.sim.model._model;data=inner.sim.data._data
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(model,data,full,spec)
        mujoco.mj_forward(model,data)
    ctrl=inner.robots[0].controller
    for key,value in payload['controller'].items():
        setattr(ctrl,key,value.copy() if isinstance(value,np.ndarray) else value)
    for key,value in payload['sim_data'].items():
        target=getattr(inner.sim.data,key)
        if isinstance(target,np.ndarray):target[...] = value


def build_env(task_id=6,trial_index=0,seed_base=7):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite=benchmark.get_benchmark_dict()['libero_10']()
    task=suite.get_task(task_id)
    language=_task_language('libero_10',task)
    initial=np.asarray(suite.get_task_init_states(task_id)[trial_index])
    bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,
                          horizon=environment_horizon(520,10))
    env.seed(seed_base+trial_index)
    env.reset()
    obs=env.set_init_state(initial)
    done=False
    for _ in range(10):
        obs,_,rd,_=env.step([0.]*6+[-1.])
        _,done=episode_status(env,rd)
        if done: break
    return env,obs,language,initial,suite,task,done


def capture(port,task_id=6,trial_index=0,seed_base=7,fractions=(.2,.4,.6),out_root=OUT):
    folder=Path(out_root)/'trials'/f'task{task_id:02d}_init{trial_index:02d}'
    cpdir=folder/'branchpoints'
    cpdir.mkdir(parents=True,exist_ok=True)
    env,obs,language,initial,suite,task,done=build_env(task_id,trial_index,seed_base)
    client=S2WebsocketClientPolicy('127.0.0.1',port)
    chunks=[]; checkpoints=[]; action_plan=[]; query_id=0; steps=0; success=False
    targets={int(round((float(f)*520)/5)*5):float(f) for f in fractions}
    started=time.monotonic()
    try:
        while steps<520 and not done:
            if not action_plan:
                if steps in targets:
                    flat=np.asarray(env.sim.get_state().flatten(),np.float64)
                    cp_path=cpdir/f'at_{int(targets[steps]*100):02d}.npz'
                    state_sha=hashlib.sha256(flat.tobytes()).hexdigest()
                    req=request(obs,language,task_id,trial_index,query_id,'normal_rank',seed_base)
                    response=client.infer(req)
                    debug=response.get('s2_debug',{})
                    np.savez_compressed(cp_path,sim_state=flat,initial_state=initial,
                        env_steps=np.asarray(steps),query_id=np.asarray(query_id),
                        observation_sha256=np.asarray(observation_sha(obs)),
                        observation_agentview=obs['agentview_image'],
                        observation_wrist=obs['robot0_eye_in_hand_image'],
                        observation_state=np.concatenate((obs['robot0_eef_pos'],obs['robot0_eef_quat'],obs['robot0_gripper_qpos'])),
                        python_sim_state=python_sim_snapshot(env),
                        task_id=np.asarray(task_id),trial_index=np.asarray(trial_index),
                        trial_seed=np.asarray(seed_base+trial_index),
                        candidates=np.asarray([x['memory_row'] for x in debug['top8']],np.int64),
                        scores=np.asarray([x['score'] for x in debug['top8']],np.float32))
                    checkpoints.append({'fraction':targets[steps],'env_steps':steps,'query_id':query_id,
                        'path':str(cp_path),'state_sha256':state_sha,'candidates':[x['memory_row'] for x in debug['top8']]})
                else:
                    req=request(obs,language,task_id,trial_index,query_id,'normal_rank',seed_base)
                    response=client.infer(req)
                    debug=response.get('s2_debug',{})
                chunk=np.asarray(response['actions'],np.float32)[:,:7]
                if len(chunk)<5: raise ValueError(f'short action chunk: {chunk.shape}')
                action_plan.extend(chunk[:5])
                chunks.append({'query_id':query_id,'seed':(seed_base+trial_index)*100000+query_id,
                    'observation_sha256':observation_sha(obs),
                    'sha256':hashlib.sha256(np.ascontiguousarray(chunk).tobytes()).hexdigest(),
                    'actions':chunk.tolist(),'debug':debug})
                query_id+=1
            obs,_,rd,_=env.step(np.asarray(action_plan.pop(0)).tolist())
            steps+=1
            success,done=episode_status(env,rd)
        result={'status':'complete','task_id':task_id,'trial_index':trial_index,'trial_seed':seed_base+trial_index,
          'initial_state_sha256':hashlib.sha256(np.ascontiguousarray(initial).view(np.uint8)).hexdigest(),
          'success':success,'env_steps':steps,'termination_reason':'success' if success else 'timeout_or_done',
          'checkpoints':checkpoints,'queries':chunks,'elapsed_seconds':time.monotonic()-started,
          'task':language}
        (folder/'normal_replay.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        return result
    finally:
        client.close();env.close()


def branch(port,checkpoint_path,condition,*,memory_row=None,t0=None,seed_offset=0,replay_path=None):
    data=np.load(checkpoint_path,allow_pickle=False)
    task_id=int(data['task_id']);trial_index=int(data['trial_index'])
    trial_seed=int(data['trial_seed']);seed_base=trial_seed-trial_index
    env,obs,language,initial,_,_,done=build_env(task_id,trial_index,seed_base)
    query_id=int(data['query_id']);start_steps=int(data['env_steps']);steps=0
    replay_state_delta=None
    if replay_path is None:
        obs=env.set_init_state(np.asarray(data['sim_state']))
        restore_python_sim_snapshot(env,data['python_sim_state'])
        obs['agentview_image']=np.asarray(data['observation_agentview']).copy()
        obs['robot0_eye_in_hand_image']=np.asarray(data['observation_wrist']).copy()
        saved_robot=np.asarray(data['observation_state'])
        obs['robot0_eef_pos']=saved_robot[:3].copy()
        obs['robot0_eef_quat']=saved_robot[3:7].copy()
        obs['robot0_gripper_qpos']=saved_robot[7:].copy()
        restored_state=np.asarray(env.sim.get_state().flatten(),np.float64)
        replay_state_delta=float(np.max(np.abs(restored_state-np.asarray(data['sim_state'],np.float64))))
        steps=start_steps
        resume_mode='flattened_state_restore'
    else:
        # MuJoCo's flattened MjSimState stores only time/qpos/qvel. Replaying
        # the recorded pre-branch actions recreates non-serialized controller
        # and solver state exactly; the saved flat state is then asserted.
        replay=json.loads(Path(replay_path).read_text())
        for chunk in replay['queries']:
            if int(chunk['query_id'])>=query_id:break
            for action in np.asarray(chunk['actions'],np.float32)[:5]:
                obs,_,rd,_=env.step(action.tolist())
                steps+=1
                _,done=episode_status(env,rd)
                if done:break
            if done:break
        if steps!=start_steps:raise ValueError(f'replay reached {steps}, expected branch step {start_steps}')
        current=np.asarray(env.sim.get_state().flatten(),np.float64)
        replay_state_delta=float(np.max(np.abs(current-np.asarray(data['sim_state'],np.float64))))
        if replay_state_delta>1e-12:raise ValueError(f'action replay state differs from checkpoint by {replay_state_delta}')
        if observation_sha(obs)!=str(data['observation_sha256']):
            raise ValueError('action replay observation differs from saved checkpoint observation')
        resume_mode='deterministic_action_replay_to_saved_checkpoint'
    client=S2WebsocketClientPolicy('127.0.0.1',port)
    action_plan=[];query_hashes=[];traces=[];success=bool(env.check_success());returned_done=False
    first_observation_sha=observation_sha(obs)
    obs_diagnostics={}
    for name,key,actual_key in [('agentview','observation_agentview','agentview_image'),
                                ('wrist','observation_wrist','robot0_eye_in_hand_image')]:
        ref=np.asarray(data[key]);actual=np.asarray(obs[actual_key])
        obs_diagnostics[name]={'max_abs':int(np.max(np.abs(actual.astype(np.int16)-ref.astype(np.int16)))),
          'different_values':int(np.count_nonzero(actual!=ref)),'shape':list(actual.shape)}
    robot_state=np.concatenate((obs['robot0_eef_pos'],obs['robot0_eef_quat'],obs['robot0_gripper_qpos']))
    ref_robot=np.asarray(data['observation_state'])
    obs_diagnostics['robot_state']={'max_abs':float(np.max(np.abs(robot_state-ref_robot))),
      'different_values':int(np.count_nonzero(robot_state!=ref_robot))}
    started=time.monotonic()
    try:
        while steps<520 and not success and not returned_done:
            if not action_plan:
                args={}
                if memory_row is not None: args['memory_row']=int(memory_row);args['branch_t0']=float(t0)
                req=request(obs,language,task_id,trial_index,query_id,condition,seed_base,**args)
                if condition=='normal_alt': req['s2_rng_seed']+=int(seed_offset)
                response=client.infer(req)
                chunk=np.asarray(response['actions'],np.float32)[:,:7]
                traces.append(response.get('s2_debug',{}))
                query_hashes.append({'query_id':query_id,'sha256':hashlib.sha256(np.ascontiguousarray(chunk).tobytes()).hexdigest()})
                action_plan.extend(chunk[:5]);query_id+=1
            obs,_,rd,_=env.step(np.asarray(action_plan.pop(0)).tolist())
            steps+=1;success,returned_done=episode_status(env,rd)
        return {'status':'complete','condition':condition,'memory_row':memory_row,'t0':t0,
          'success':success,'env_steps':steps,'branch_start_steps':start_steps,
          'resume_mode':resume_mode,'checkpoint_sim_state_max_abs':replay_state_delta,
          'branch_observation_sha256':first_observation_sha,
          'saved_observation_sha256':str(data['observation_sha256']),
          'observation_component_differences':obs_diagnostics,
          'query_hashes':query_hashes,'traces':traces,'elapsed_seconds':time.monotonic()-started}
    finally:
        client.close();env.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['capture','branch'])
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--task-id',type=int,default=6)
    parser.add_argument('--trial-index',type=int,default=0)
    parser.add_argument('--seed-base',type=int,default=7)
    parser.add_argument('--fractions',type=float,nargs='*',default=[.2,.4,.6])
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--checkpoint-path',type=Path)
    parser.add_argument('--replay-path',type=Path)
    parser.add_argument('--condition',default='normal')
    parser.add_argument('--memory-row',type=int)
    parser.add_argument('--t0',type=float)
    parser.add_argument('--seed-offset',type=int,default=0)
    parser.add_argument('--output',type=Path)
    a=parser.parse_args()
    result=capture(a.port,a.task_id,a.trial_index,a.seed_base,a.fractions,a.output_dir) if a.mode=='capture' else branch(a.port,a.checkpoint_path,a.condition,
        memory_row=a.memory_row,t0=a.t0,seed_offset=a.seed_offset,replay_path=a.replay_path)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
