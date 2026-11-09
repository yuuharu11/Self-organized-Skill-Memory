#!/usr/bin/env python3
"""Import/version, target registry and optional EGL rendering smoke test."""
import argparse, json, pathlib

def main():
    p=argparse.ArgumentParser(); p.add_argument('--render',action='store_true'); p.add_argument('--output',default='robocasa-smoke'); a=p.parse_args()
    import robocasa, robosuite, mujoco, numpy as np
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    print('robocasa',getattr(robocasa,'__version__','1.0.1'),'robosuite',getattr(robosuite,'__version__','unknown'),'mujoco',mujoco.__version__,'numpy',np.__version__)
    for k,n in [('atomic_seen',18),('composite_seen',16),('composite_unseen',16)]:
        assert len(TASK_SET_REGISTRY[k])==n,(k,len(TASK_SET_REGISTRY[k]))
    print('target task counts: atomic_seen=18 composite_seen=16 composite_unseen=16')
    if not a.render: return
    import gymnasium as gym
    out=pathlib.Path(a.output); out.mkdir(parents=True,exist_ok=True)
    for task in ('OpenDrawer',TASK_SET_REGISTRY['composite_unseen'][0]):
        # RoboCasa's Gym wrapper sets these robosuite options itself; passing
        # them again would produce duplicate keyword arguments.
        env=gym.make('robocasa/'+task,split='target',seed=11)
        try:
            obs,_=env.reset(seed=11)
            for i in range(2):
                obs,_,term,trunc,_=env.step(env.action_space.sample())
                if term or trunc: break
            keys=('video.robot0_agentview_left','video.robot0_agentview_right','video.robot0_eye_in_hand')
            for key in keys:
                image=np.asarray(obs[key]); assert image.ndim==3 and image.shape[-1] in (3,4),(key,image.shape)
                from PIL import Image
                Image.fromarray(image[...,:3].astype('uint8')).save(out/f'{task}_{key.rsplit("_",1)[-1]}.png')
            print(task,'camera shapes',*[np.asarray(obs[k]).shape for k in keys])
        finally: env.close()
if __name__=='__main__': main()
