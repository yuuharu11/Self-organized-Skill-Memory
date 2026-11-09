"""RoboCasa target-split websocket evaluation client (run in RoboCasa venv)."""
import argparse, json, pathlib, time, subprocess
import numpy as np
from openpi_client import image_tools, websocket_client_policy
import gymnasium as gym
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action

def main():
    p=argparse.ArgumentParser(); p.add_argument('--host',default='127.0.0.1'); p.add_argument('--port',type=int,default=8000); p.add_argument('--task-set',default='atomic_seen'); p.add_argument('--tasks',nargs='*'); p.add_argument('--split',default='target'); p.add_argument('--num-trials',type=int,default=1); p.add_argument('--replan-steps',type=int,default=5); p.add_argument('--output'); p.add_argument('--seed',type=int,default=7); p.add_argument('--replay-demo-dir'); p.add_argument('--dataset-root'); p.add_argument('--replay-episode',type=int,default=0); a=p.parse_args()
    tasks=a.tasks or TASK_SET_REGISTRY[a.task_set]
    out=pathlib.Path(a.output or pathlib.Path(__import__('os').environ.get('SKILLNET_STATE_DIR','.'))/'outputs'/f"robocasa-eval-{int(time.time())}"); out.mkdir(parents=True,exist_ok=False)
    client=websocket_client_policy.WebsocketClientPolicy(a.host,a.port,keepalive=False); records=[]  # no pings: the first pi0.5 call JIT-compiles for minutes
    repo=pathlib.Path(__file__).resolve().parents[3]
    try: commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    except Exception: commit='unknown'
    for task in tasks:
        # RoboCasaGymEnv fixes offscreen cameras and render flags internally.
        env=gym.make('robocasa/'+task,split=a.split,seed=a.seed)
        successes=0
        try:
            for trial in range(a.num_trials):
                if a.replay_demo_dir:
                    import gzip
                    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
                    # Satisfy Gym's OrderEnforcing wrapper before directly
                    # restoring the underlying simulator state.
                    env.reset(seed=a.seed+trial)
                    demo=pathlib.Path(a.replay_demo_dir); states=np.load(demo/'states.npz')['states']
                    meta=json.loads((demo/'ep_meta.json').read_text())
                    with gzip.open(demo/'model.xml.gz','rt') as f: xml=f.read()
                    gym_env=env.unwrapped
                    sim_env=gym_env.env
                    reset_to(sim_env,{'model':xml,'states':states[0],'ep_meta':json.dumps(meta)})
                    # RoboCasa's upstream reset_to mutates the simulator but
                    # intentionally returns None. Recompute the observation.
                    obs=gym_env.get_observation(sim_env._get_observations(force_update=True))
                else: obs,_=env.reset(seed=a.seed+trial)
                queue=[]; success=False; frames=[]
                for step in range(get_task_horizon(task)):
                    if not queue:
                        q=lambda name:image_tools.convert_to_uint8(image_tools.resize_with_pad(np.asarray(obs[name]),224,224))
                        # Keep the 16-D ordering used by the converted dataset modality.
                        state=np.concatenate([obs[k] for k in ('state.base_position','state.base_rotation','state.end_effector_position_relative','state.end_effector_rotation_relative','state.gripper_qpos')]).astype(np.float32)
                        assert state.shape==(16,),state.shape
                        result=client.infer({'observation/image':q('video.robot0_agentview_left'),'observation/right_image':q('video.robot0_agentview_right'),'observation/wrist_image':q('video.robot0_eye_in_hand'),'observation/state':state,'prompt':obs['annotation.human.task_description']})
                        queue=list(np.asarray(result['actions'])[:a.replan_steps])
                    obs,_,_,_,info=env.step(convert_action(np.asarray(queue.pop(0))))
                    if a.output and 'video.robot0_agentview_left' in obs:
                        frames.append(np.asarray(obs['video.robot0_agentview_left']))
                    if info.get('success',False): success=True; break
                if frames:
                    import imageio.v2 as imageio
                    imageio.mimsave(out/f'{task}_trial{trial:03d}.mp4',frames,fps=20)
                successes+=int(success); records.append({'task':task,'trial':trial,'success':success,'steps':step+1})
        finally: env.close()
        print(task,successes,'/',a.num_trials)
    report={'split':a.split,'task_set':a.task_set,'tasks':tasks,'num_trials':a.num_trials,'seed':a.seed,'policy':f'ws://{a.host}:{a.port}','records':records,'success_rate':sum(x['success'] for x in records)/max(1,len(records)),'config':vars(a),'commit_id':commit}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n'); print('output',out)
if __name__=='__main__': main()
