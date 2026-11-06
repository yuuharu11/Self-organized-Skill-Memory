"""Prepare fixed unseen-task demonstration windows and VLM prefix features."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

import h5py
import numpy as np

HERE=Path(__file__).resolve().parent
REPO_ROOT=Path(__file__).resolve().parents[5]
STATE_DIR=Path(os.environ.get('SKILLNET_STATE_DIR',REPO_ROOT/'.skillnet-state')).expanduser()
ROOT=Path(os.environ.get('SKILLNET_PI05_ROOT',STATE_DIR/'outputs/memory-pilot/pi05-memory40-20261104')).expanduser()
RUN=ROOT/'run-01'
DEFAULT_OUT=Path(os.environ.get('UNSEEN_LIBERO_OUT',STATE_DIR/'outputs/memory-pilot/unseen-libero/oracle-01')).expanduser()
DEFAULT_H5=Path(os.environ.get('SKILLNET_LIBERO_HDF5',STATE_DIR/'data/LIBERO')).expanduser()
CHECKPOINT=Path(os.environ.get('SKILLNET_PI05_CHECKPOINT',STATE_DIR/'checkpoints/openpi-assets/checkpoints/pi05_libero')).expanduser()

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)

def fixed_tasks():
    chosen=json.loads((STATE_DIR/'outputs/memory-pilot/unseen-libero/baseline-01/decision.json').read_text())
    evaluation=[{'task_id':int(t['task_id']),'name':t['name'],'instruction':t['instruction'],'split':'eval'} for t in chosen['tasks']]
    from libero.libero import benchmark
    suite=benchmark.get_benchmark_dict()['libero_90']()
    dev=[]
    for i in (6,7):
        t=suite.get_task(i);dev.append({'task_id':i,'name':t.name,'instruction':t.language,'split':'dev'})
    if {x['task_id'] for x in evaluation}&{x['task_id'] for x in dev}:raise AssertionError('Dev/Eval task overlap')
    return dev,evaluation

def prepare(out:Path,h5root:Path,checkpoint:Path):
    if (out/'unseen_rows/manifest.json').exists():raise FileExistsError('prepared dataset already exists; resume is read-only')
    import pyarrow
    from PIL import Image
    from openpi.transforms import Normalize,ResizeImages,TokenizePrompt,PadStatesAndActions
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.models.model import ModelType
    from openpi.shared.normalize import load
    dev,ev=fixed_tasks();tasks=dev+ev
    norm_files=list((checkpoint/'assets').rglob('norm_stats.json'))
    if len(norm_files)!=1:raise ValueError(f'expected single norm stats file, found {norm_files}')
    norm_path=norm_files[0]
    run_manifest=json.loads((RUN/'data_manifest.json').read_text())
    run_instructions=set(run_manifest['tasks'].values())
    for item in tasks:
        if item['instruction'] in run_instructions:raise AssertionError('unseen task instruction leaked into LIBERO-40')
        h5path=h5root/(item['name']+'_demo.hdf5')
        if not h5path.is_file():raise FileNotFoundError(h5path)
        item['hdf5']=str(h5path)
    inputs=LiberoInputs(ModelType.PI05)
    transforms=[Normalize(load(norm_path.parent),use_quantiles=True),ResizeImages(224,224),
                TokenizePrompt(PaligemmaTokenizer(200),discrete_state_input=False),PadStatesAndActions(32)]
    records=[];rows=[]
    for task in tasks:
        with h5py.File(task['hdf5'],'r') as f:
            root=f['data']; demos=sorted([x for x in root.keys() if x.startswith('demo_')],key=lambda s:int(s.split('_')[1]))
            if len(demos)<10:raise ValueError(f"{task['name']} has only {len(demos)} demos")
            for epi,name in enumerate(demos[:10]):
                d=root[name];obs=d['obs'];actions=np.asarray(d['actions'],np.float32)[::2]
                ee=np.asarray(obs['ee_pos'],np.float32)[::2];ori=np.asarray(obs['ee_ori'],np.float32)[::2]
                grip=np.asarray(obs['gripper_states'],np.float32)[::2]
                base=np.asarray(obs['agentview_rgb'],np.uint8)[::2];wrist=np.asarray(obs['eye_in_hand_rgb'],np.uint8)[::2]
                n=min(map(len,(actions,ee,ori,grip,base,wrist)))
                actions,ee,ori,grip,base,wrist=[x[:n] for x in (actions,ee,ori,grip,base,wrist)]
                if ee.shape[1]!=3 or ori.shape[1]!=3 or grip.shape[1]!=2 or actions.shape[1]!=7:raise ValueError('HDF5 observation/action schema changed')
                state=np.concatenate([ee,ori,grip],axis=1).astype(np.float32)
                count=0
                for start in range(0,n-9,10):
                    # Preserve the public LIBERO input transform exactly; the
                    # HDF5 source images are 128x128 and are padded to 224 by
                    # ResizeImages, while actions/state use pi05_libero stats.
                    x={'observation/image':base[start],'observation/wrist_image':wrist[start],
                       'observation/state':state[start],'actions':actions[start:start+10],
                       'prompt':task['instruction']}
                    x=inputs(x)
                    for fn in transforms:x=fn(x)
                    rows.append({'image_base':x['image']['base_0_rgb'],'image_wrist':x['image']['left_wrist_0_rgb'],
                                 'state':x['state'],'actions':x['actions'],'prompt':x['tokenized_prompt'],
                                 'prompt_mask':x['tokenized_prompt_mask'],'task':task['task_id'],'episode':epi,
                                 'frame':start,'split':task['split'],'name':task['name'],'instruction':task['instruction']})
                    count+=1
                records.append({'task_id':task['task_id'],'task_name':task['name'],'instruction':task['instruction'],
                                'split':task['split'],'episode':epi,'source_demo':name,'source_hdf5':task['hdf5'],
                                'source_steps':int(n),'rows':int(count),'subsample':'every other HDF5 frame (10Hz)'} )
    if not rows:raise ValueError('no windows created')
    rd=out/'unseen_rows';rd.mkdir(parents=True,exist_ok=False)
    fields={'image_base':(np.uint8,(224,224,3)),'image_wrist':(np.uint8,(224,224,3)),
            'state':(np.float32,(32,)),'actions':(np.float32,(10,32)),
            'prompt':(np.int32,(200,)),'prompt_mask':(bool,(200,)),'task':(np.int32,()),
            'episode':(np.int32,()),'frame':(np.int32,())}
    for key,(dtype,tail) in fields.items():
        arr=np.lib.format.open_memmap(rd/(key+'.npy'),mode='w+',dtype=dtype,shape=(len(rows),)+tail)
        for i,r in enumerate(rows):arr[i]=r[key]
        arr.flush()
    split=np.asarray([r['split']=='dev' for r in rows],bool)
    task_name=np.asarray([r['name'] for r in rows],dtype='U150'); instruction=np.asarray([r['instruction'] for r in rows],dtype='U300')
    np.save(rd/'is_dev.npy',split);np.save(rd/'task_name.npy',task_name);np.save(rd/'instruction.npy',instruction)
    write_json(rd/'manifest.json',{'rows':len(rows),'episodes':len(records),'dev_tasks':[x['task_id'] for x in dev],
        'eval_tasks':[x['task_id'] for x in ev],'dev_episodes_each':10,'eval_episodes_each':10,
        'windows':'10 action frames, stride 10, source 10Hz (HDF5 every other frame)',
        'actions_normalized_with':str(norm_path),'norm_sha256':hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        'source_records':records,'source_note':'held-out demonstrations only; excluded from LIBERO-40 memory and Expert gradient updates'})
    print(json.dumps({'stage':'prepare','rows':len(rows),'dev_rows':int(split.sum()),'eval_rows':int((~split).sum())}),flush=True)

def extract(out:Path,checkpoint:Path,batch=4):
    started=time.monotonic()
    import flax.nnx as nnx
    import jax
    from openpi.shared.nnx_utils import module_jit
    base=module(HERE.parent/'pi05_memory'/'run.py','unseen_oracle_run_base')
    rows={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in (out/'unseen_rows').glob('*.npy')}
    n=len(rows['actions']);model,_=base.load_policy(checkpoint);encode=module_jit(model.prefix_features)
    tok=np.lib.format.open_memmap(out/'unseen_rows/tokens.npy',mode='w+',dtype='float16',shape=(n,712,2048))
    grp=np.lib.format.open_memmap(out/'unseen_rows/groups.npy',mode='w+',dtype='int8',shape=(n,712))
    pooled=np.lib.format.open_memmap(out/'unseen_rows/pooled.npy',mode='w+',dtype='float32',shape=(n,2048))
    for st in range(0,n,batch):
        ids=np.arange(st,min(st+batch,n));obs=base.obs_batch(rows,ids);h,g=encode(obs);h,g=np.asarray(h),np.asarray(g)
        tok[ids]=h.astype(np.float16);grp[ids]=g.astype(np.int8)
        valid=g>=0;pooled[ids]=(h*valid[...,None]).sum(1)/np.maximum(valid.sum(1),1)[:,None]
        if st%(batch*100)==0:
            tok.flush();grp.flush();pooled.flush()
            write_json(out/'unseen_rows/feature_progress.json',{'rows':min(st+batch,n),'total':n,'seconds_elapsed':time.monotonic()-started})
    tok.flush();grp.flush();pooled.flush()
    write_json(out/'unseen_rows/features.json',{'rows':n,'shape_tokens':[712,2048],'checkpoint':str(checkpoint),'complete':True})
    print(json.dumps({'stage':'extract','rows':n}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract']);p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--hdf5-root',type=Path,default=DEFAULT_H5);p.add_argument('--checkpoint',type=Path,default=CHECKPOINT);p.add_argument('--batch',type=int,default=4);a=p.parse_args()
    if a.stage=='prepare':prepare(a.out,a.hdf5_root,a.checkpoint)
    else:extract(a.out,a.checkpoint,a.batch)
if __name__=='__main__':main()
