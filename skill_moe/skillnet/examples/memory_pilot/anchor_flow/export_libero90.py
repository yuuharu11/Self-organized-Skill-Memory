#!/usr/bin/env python3
"""Export all LIBERO-90 RLDS demonstrations into checkpoint-normalized S2 rows.

Use nonoverlapping complete H=10 chunks. Text is the episode instruction;
the separate frozen SkillNet routing input uses only the task-level 90 labels.
No annotation text or annotated segment boundaries enter the prefix prompt.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def complete_starts(length,horizon=10,stride=10):
    return np.arange(0,max(length-horizon+1,0),stride,dtype=np.int64)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rlds',type=Path,default=Path('/work/.skillnet-state/data/libero/libero90_rlds/libero_90_openvla_processed/1.0.0'))
    parser.add_argument('--raw90',type=Path,default=Path('/work/data/LIBERO/libero_90'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stride',type=int,default=10)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.stride<1: raise ValueError('stride must be positive')
    # This preprocessing process is CPU-only and never takes GPU memory.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    from openpi import transforms
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import normalize
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.models.model import ModelType
    from smoke import DEFAULT_CHECKPOINT

    here=Path(__file__).resolve()
    annotations=json.loads((here.parents[2]/'libero/annotations/instruct2plan_obj_90.json').read_text())
    allowed_files={p.name for p in args.raw90.glob('*.hdf5')}
    if len(allowed_files)!=90: raise ValueError('expected exactly 90 raw LIBERO-90 task files')
    norm_file=Path(DEFAULT_CHECKPOINT).parent/'assets/jsw19/libero_90_v1/norm_stats.json'
    norm_sha=hashlib.sha256(norm_file.read_bytes()).hexdigest()
    stats=normalize.load(norm_file.parent)
    tokenizer=PaligemmaTokenizer(200)
    process=transforms.compose([
        LiberoInputs(ModelType.PI05),transforms.Normalize(stats,use_quantiles=True),
        transforms.ResizeImages(224,224),transforms.TokenizePrompt(tokenizer,discrete_state_input=False),
        transforms.PadStatesAndActions(32)])
    files=sorted(args.rlds.glob('*tfrecord*'))
    if not files: raise FileNotFoundError(args.rlds)
    cols=defaultdict(list)
    task_counts=defaultdict(int)
    for episode_index,serialized in enumerate(tf.data.TFRecordDataset([str(p) for p in files],num_parallel_reads=1)):
        features=tf.train.Example.FromString(serialized.numpy()).features.feature
        original=features['episode_metadata/file_path'].bytes_list.value[0].decode()
        task_file=Path(original).name
        if task_file not in allowed_files: raise ValueError(f'non-LIBERO-90 source file {original}')
        actions=np.asarray(features['steps/action'].float_list.value,np.float32).reshape(-1,7)
        states=np.asarray(features['steps/observation/state'].float_list.value,np.float32).reshape(-1,8)
        language=[v.decode() for v in features['steps/language_instruction'].bytes_list.value]
        if len(set(language))!=1: raise ValueError('instruction changes within an episode')
        instruction=language[0]
        route=annotations[instruction]['all_classes']
        images=features['steps/observation/image'].bytes_list.value
        wrists=features['steps/observation/wrist_image'].bytes_list.value
        if not(len(actions)==len(states)==len(images)==len(wrists)==len(language)):
            raise ValueError('RLDS arrays have different step counts')
        # Original demo/init indices are absent from this RLDS export. Group
        # ALL recordings of the task to conservatively exclude nearby sources.
        init_group='conservative-task-group:'+task_file
        episode_id=f'libero90/{task_file}/rlds-{episode_index:06d}'
        for start in complete_starts(len(actions),stride=args.stride):
            with Image.open(io.BytesIO(images[start])) as im: base=np.asarray(im.convert('RGB'))
            with Image.open(io.BytesIO(wrists[start])) as im: wrist=np.asarray(im.convert('RGB'))
            row=process({'observation/image':base,'observation/wrist_image':wrist,
                         'observation/state':states[start].copy(),'actions':actions[start:start+10].copy(),
                         'prompt':instruction,'skills':str(route)})
            for name in ('state','actions','skills','skill_mask','tokenized_prompt','tokenized_prompt_mask'):
                cols[name].append(row[name])
            cols['image_base'].append(row['image']['base_0_rgb'])
            cols['image_left_wrist'].append(row['image']['left_wrist_0_rgb'])
            for name,value in [('episode_id',episode_id),('init_state_id',init_group),('suite','libero90'),
                               ('instruction',instruction),('source_file',task_file),('frame',int(start))]:
                cols[name].append(value)
            task_counts[task_file]+=1
        if episode_index%100==0: print(json.dumps({'episodes':episode_index+1,'rows':len(cols['actions'])}),flush=True)
    data={name:np.asarray(values) for name,values in cols.items()}
    n=len(data['actions'])
    data.update(key_text_source=np.asarray('instruction_only'),normalization_sha256=np.asarray(norm_sha),
                source_dataset=np.asarray(str(args.rlds.resolve())),
                image_mask_base_0_rgb=np.ones(n,bool),image_mask_left_wrist_0_rgb=np.ones(n,bool),
                image_mask_right_wrist_0_rgb=np.zeros(n,bool),
                initialization_grouping=np.asarray('all recordings of same original task file'),
                checkpoint=np.asarray(DEFAULT_CHECKPOINT),sampling_stride=np.asarray(args.stride))
    if len(task_counts)!=90: raise ValueError(f'only {len(task_counts)} task sources in RLDS')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:np.savez_compressed(stream,**data)
    report={'rows':n,'episodes':episode_index+1,'tasks':len(task_counts),'stride':args.stride,
            'horizon':10,'no_incomplete_chunks':True,'source':str(args.rlds.resolve()),
            'allowed_source_files':sorted(allowed_files),'key_text_source':'instruction_only',
            'normalization_file':str(norm_file),'normalization_sha256':norm_sha,
            'initialization_grouping':'conservative task-wide exclusion; not claimed true init IDs'}
    with args.output.with_suffix('.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
