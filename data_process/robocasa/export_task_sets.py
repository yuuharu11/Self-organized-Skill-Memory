#!/usr/bin/env python3
"""Export task registry metadata from the pinned RoboCasa checkout."""
import argparse, json, os, pathlib, subprocess

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='data_process/robocasa/task_sets.json'); a=p.parse_args()
    import robocasa
    source=pathlib.Path(robocasa.__file__).resolve().parent.parent
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    expected='3443cdf3dd20f9ae8ed41adba442059b506891d0'
    if commit!=expected:raise RuntimeError(f'Expected RoboCasa {expected}, imported checkout is {commit} ({source})')
    changed=subprocess.run(['git','-C',str(source),'diff','--quiet','HEAD','--'],check=False).returncode
    staged=subprocess.run(['git','-C',str(source),'diff','--cached','--quiet'],check=False).returncode
    if changed or staged:raise RuntimeError(f'Expected unmodified tracked files at the pinned RoboCasa commit: {source}')
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon, get_ds_meta
    sets={k:list(TASK_SET_REGISTRY[k]) for k in ('atomic_seen','composite_seen','composite_unseen','target50')}
    assert tuple(map(len,(sets['atomic_seen'],sets['composite_seen'],sets['composite_unseen'])))==(18,16,16)
    tasks={}
    for names in sets.values():
        for name in names:
            if name not in tasks:
                meta=get_ds_meta(name,'target','human')
                base=pathlib.Path(os.environ.get('SKILLNET_STATE_DIR','.'))/'data'/'robocasa'
                tasks[name]={'horizon':int(get_task_horizon(name)), 'dataset_path':os.path.relpath(meta['path'],base), 'split':'target'}
    out={'robocasa_commit':expected,'task_sets':sets,'tasks':tasks,
         'action_fields':{'dataset_order':{'base_motion':[0,4],'control_mode':[4,5],'end_effector_position':[5,8],'end_effector_rotation':[8,11],'gripper_close':[11,12]},'policy_command_order':{'end_effector_position':[0,3],'end_effector_rotation':[3,6],'gripper_close':[6,7],'base_motion':[7,11],'control_mode':[11,12]}},
         'cameras':['robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand']}
    pathlib.Path(a.output).parent.mkdir(parents=True,exist_ok=True); pathlib.Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
    print(f"Wrote {a.output}: {len(tasks)} tasks, sets { {k:len(v) for k,v in sets.items()} }")
if __name__=='__main__': main()
