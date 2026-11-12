"""Offline checkpoint/data transform parity probe on pretrain demonstrations."""
import argparse
import dataclasses
import json
import pathlib

import imageio.v2 as imageio
import jax
import numpy as np
import pyarrow.parquet as pq

from openpi import transforms
from openpi.models import model as model_api
from openpi.policies import policy as policy_lib
from openpi.policies.robocasa_policy import (
    configure_robocasa_subtask_generation,
    dataset_action_to_policy,
    dataset_state_to_policy,
)
from openpi.shared import normalize
from openpi.training import checkpoints, config


TASKS = ["PickPlaceCounterToCabinet", "PickPlaceSinkToCounter", "OpenDrawer",
         "CloseFridge", "TurnOnSinkFaucet", "TurnOffStove"]
DATA_ROOT = pathlib.Path("/work/.skillnet-state/data/robocasa/v1.0/pretrain/atomic")


def _mean_std(actions, stats):
    return (actions - np.asarray(stats.mean)[:actions.shape[-1]]) / (np.asarray(stats.std)[:actions.shape[-1]] + 1e-6)


def main():
    # RoboCasa follows the fork's task-only prompt unless explicitly opted in.
    subtask_generation_enabled = configure_robocasa_subtask_generation()
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=pathlib.Path,required=True)
    p.add_argument('--per-task',type=int,default=8)
    p.add_argument('--stride',type=int,default=20)
    p.add_argument('--output',type=pathlib.Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    stats=normalize.load(a.checkpoint/'assets')
    base_cfg=config.get_config('pi05_robocasa_multitask')
    params=model_api.restore_params(a.checkpoint/'params',dtype=jax.numpy.bfloat16)
    model=base_cfg.model.load(params)

    def make_policy(discrete):
        cfg=dataclasses.replace(base_cfg,model=dataclasses.replace(base_cfg.model,discrete_state_input=discrete))
        dc=cfg.data.create(cfg.assets_dirs,cfg.model)
        return policy_lib.Policy(model,transforms=[
            transforms.InjectDefaultPrompt(None),*dc.data_transforms.inputs,
            transforms.Normalize(stats,use_quantiles=False),*dc.model_transforms.inputs,
        ],output_transforms=[*dc.model_transforms.outputs,
            transforms.Unnormalize(stats,use_quantiles=False),*dc.data_transforms.outputs])
    policies={True:make_policy(True),False:make_policy(False)}
    accum={d:{k:{'full':[],'first5':[]} for k in ('correct_action_order','saved_action_order')} for d in policies}
    details=[]
    for task in TASKS:
        root=DATA_ROOT/task/'20250819'/'lerobot'
        task_rows=[json.loads(x) for x in (root/'meta/tasks.jsonl').read_text().splitlines() if x]
        selected=[]
        for parquet in sorted((root/'data').glob('**/episode_*.parquet')):
            episode=int(parquet.stem.rsplit('_',1)[1])
            rows=pq.read_table(parquet).to_pylist()
            frames=list(range(0,max(0,len(rows)-49),a.stride))
            for frame in frames:
                selected.append((episode,rows,frame))
                if len(selected)>=a.per_task: break
            if len(selected)>=a.per_task: break
        if len(selected)<a.per_task:
            raise ValueError(f"{task}: found only {len(selected)} usable query rows, requested {a.per_task}")
        task_scores={d:{k:{'full':[],'first5':[]} for k in ('correct_action_order','saved_action_order')} for d in policies}
        readers={}
        try:
            for episode,rows,frame in selected:
                if episode not in readers:
                    readers[episode]={name:imageio.get_reader(root/'videos/chunk-000'/
                        f'observation.images.{name}'/f'episode_{episode:06d}.mp4')
                        for name in ('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand')}
                cameras=readers[episode]
                row=rows[frame]
                state=dataset_state_to_policy(np.asarray(row['observation.state'],np.float32))
                q=lambda cam:cameras[cam].get_data(frame)
                task_idx=int(row['annotation.human.task_description'])
                obs={
                    'observation/image':q('robot0_agentview_left'),
                    'observation/right_image':q('robot0_agentview_right'),
                    'observation/wrist_image':q('robot0_eye_in_hand'),
                    'observation/state':state,
                    'prompt':task_rows[task_idx]['task'],
                }
                target_raw=np.stack([np.asarray(rows[t]['action'],np.float32) for t in range(frame,frame+50)])
                target_correct=dataset_action_to_policy(target_raw)
                target_correct_norm=_mean_std(target_correct,stats['actions'])
                target_saved_norm=_mean_std(target_raw,stats['actions'])
                for discrete,policy in policies.items():
                    pred=np.asarray(policy.infer(obs)['actions'],np.float32)
                    pred_norm=_mean_std(pred,stats['actions'])
                    for name,target in [('correct_action_order',target_correct_norm),('saved_action_order',target_saved_norm)]:
                        err=(pred_norm-target)**2
                        vals={'full':float(err.mean()),'first5':float(err[:5].mean())}
                        for span,val in vals.items(): task_scores[discrete][name][span].append(val)
                    details.append({'task':task,'frame':frame,'discrete_state_input':discrete,
                                    'mse_true_order_full':float(np.mean((pred_norm-target_correct_norm)**2)),
                                    'mse_true_order_first5':float(np.mean((pred_norm[:5]-target_correct_norm[:5])**2)),
                                    'mse_saved_order_full':float(np.mean((pred_norm-target_saved_norm)**2)),
                                    'mse_saved_order_first5':float(np.mean((pred_norm[:5]-target_saved_norm[:5])**2))})
            for discrete in policies:
                for order in task_scores[discrete]:
                    for span,values in task_scores[discrete][order].items(): accum[discrete][order][span].extend(values)
            print(task, 'queries', len(selected), 'episodes', sorted(readers),
                  {str(d):{o:{k:round(float(np.mean(v)),6) for k,v in spans.items()}
                           for o,spans in scores.items()} for d,scores in task_scores.items()},flush=True)
        finally:
            for cameras in readers.values():
                for r in cameras.values(): r.close()
    summary={}
    for d,scores in accum.items():
        summary[str(d)]={o:{span:{'mse':float(np.mean(vals)),'n':len(vals)} for span,vals in spans.items()}
                         for o,spans in scores.items()}
    payload={'tasks':TASKS,'frames_per_task':a.per_task,'total_frames':len(details),
             'action_metric':'normalized checkpoint action space, 12 command dimensions',
             'subtask_generation_enabled':subtask_generation_enabled,
             'summary':summary,'rows':details}
    a.output.write_text(json.dumps(payload,indent=2)+'\n')
    print('PARITY',json.dumps(summary,sort_keys=True),flush=True)


if __name__=='__main__': main()
