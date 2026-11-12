"""Build and feature LIBERO-90 transfer rows for X1.

The row transform is intentionally copied by composition from prepare_oracle's
public helpers (same image orientation and pi05 input transforms).  The 78
non-Dev/non-Eval tasks are split by task, never by row.  All files are
append/resume safe; a completed cache is immutable.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

import h5py
import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import prepare_oracle as po
from unseen_paths import orient_libero_frame

STATE=Path(os.environ.get('SKILLNET_STATE_DIR',Path(__file__).resolve().parents[5]/'.skillnet-state'))
CHECKPOINT=po.CHECKPOINT
H5ROOT=Path(os.environ.get('SKILLNET_LIBERO_HDF5',STATE/'data/libero/datasets'))/'libero_90'
EVAL_TASKS=(0,10,13,31,35,43,57,61,84,88)
DEV_TASKS=(6,7)
SEED=4101

def task_table():
    from libero.libero import benchmark
    suite=benchmark.get_benchmark_dict()['libero_90']()
    return [{'task_id':i,'name':suite.get_task(i).name,'instruction':suite.get_task(i).language}
            for i in range(int(suite.n_tasks))]

def choose_tasks(*,smoke=False):
    rows=[x for x in task_table() if x['task_id'] not in set(EVAL_TASKS+DEV_TASKS)]
    rng=np.random.default_rng(SEED)
    if smoke:
        # Four fixed tasks: two fit and two selection. Keep the exact ids in manifest.
        chosen=rows[:4]
        fit=[x['task_id'] for x in chosen[:2]]; select=[x['task_id'] for x in chosen[2:]]
        return chosen,fit,select
    ids=np.array([x['task_id'] for x in rows]); select=sorted(rng.choice(ids,size=16,replace=False).tolist())
    return rows,[int(i) for i in ids if i not in select],select

def _write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)

def active_action_channels(padded_actions):
    """Return the seven active LIBERO action channels before right padding."""
    a=np.asarray(padded_actions,np.float32)
    if a.ndim<1 or a.shape[-1]<7:raise ValueError('padded action width must be at least seven')
    return a[..., :7]

def _transformers():
    from openpi.transforms import Normalize,ResizeImages,TokenizePrompt,PadStatesAndActions
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.models.model import ModelType
    from openpi.shared.normalize import load
    norms=list((CHECKPOINT/'assets').rglob('norm_stats.json'))
    if len(norms)!=1: raise FileNotFoundError(f'expected one pi05 norm_stats.json, got {norms}')
    trans=[Normalize(load(norms[0].parent),use_quantiles=True),ResizeImages(224,224),
           TokenizePrompt(PaligemmaTokenizer(200),discrete_state_input=False),PadStatesAndActions(32)]
    return LiberoInputs(ModelType.PI05),trans,norms[0]

def prepare(out,*,smoke=False,demos=20,h5root=H5ROOT):
    out=Path(out)
    if out.exists(): raise FileExistsError(f'{out} exists; choose a fresh output directory')
    from PIL import Image  # imported intentionally with the image-transform dependencies
    tasks,fit_tasks,select_tasks=choose_tasks(smoke=smoke)
    inputs,trans,norm_path=_transformers();rows=[];episodes=[]
    for task in tasks:
        h5path=Path(h5root)/(task['name']+'_demo.hdf5')
        if not h5path.is_file(): raise FileNotFoundError(h5path)
        with h5py.File(h5path,'r') as f:
            names=sorted((x for x in f['data'] if x.startswith('demo_')),key=lambda x:int(x.split('_')[1]))
            n_demo=min(int(demos),len(names))
            for epi,name in enumerate(names[:n_demo]):
                d=f['data'][name];obs=d['obs'];a=np.asarray(d['actions'],np.float32)[::2]
                pos=np.asarray(obs['ee_pos'],np.float32)[::2];ori=np.asarray(obs['ee_ori'],np.float32)[::2]
                grip=np.asarray(obs['gripper_states'],np.float32)[::2]
                base=np.asarray(obs['agentview_rgb'],np.uint8)[::2];wrist=np.asarray(obs['eye_in_hand_rgb'],np.uint8)[::2]
                n=min(map(len,(a,pos,ori,grip,base,wrist)));a,pos,ori,grip,base,wrist=[x[:n] for x in (a,pos,ori,grip,base,wrist)]
                state=np.concatenate([pos,ori,grip],axis=1).astype(np.float32);nr=0
                for start in range(0,n-9,10):
                    prev=a[max(0,start-10):start]
                    # Normalize executed history using the exact action transform, then zero-pad missing prefix.
                    x={'observation/image':orient_libero_frame(base[start]),
                       'observation/wrist_image':orient_libero_frame(wrist[start]),
                       'observation/state':state[start], 'actions':a[start:start+10], 'prompt':task['instruction']}
                    x=inputs(x)
                    for fn in trans:x=fn(x)
                    hp=np.zeros((10,7),np.float32)
                    if len(prev):
                        hx={'observation/image':orient_libero_frame(base[start]),
                            'observation/wrist_image':orient_libero_frame(wrist[start]),
                            'observation/state':state[start], 'actions':prev, 'prompt':task['instruction']}
                        hx=inputs(hx)
                        # Use the same normalizer and action padding transforms, without changing the stored row.
                        for fn in trans[:1]+trans[3:]:hx=fn(hx)
                        hnorm=active_action_channels(hx['actions'])
                        hp[-min(10,len(hnorm)):]=hnorm[-10:]
                    task_split='select' if task['task_id'] in select_tasks else 'fit'
                    rows.append({'image_base':x['image']['base_0_rgb'],'image_wrist':x['image']['left_wrist_0_rgb'],
                        'state':x['state'],'actions':x['actions'],'hist':hp,'has_hist':bool(start>0),
                        'prompt':x['tokenized_prompt'],'prompt_mask':x['tokenized_prompt_mask'],
                        'task':int(task['task_id']),'episode':int(epi),'frame':int(start),
                        'is_select':bool(task_split=='select')})
                    nr+=1
                episodes.append({'task_id':task['task_id'],'task_name':task['name'],'instruction':task['instruction'],
                    'episode':epi,'demo':name,'rows':nr,'source_steps':int(n),'hdf5':str(h5path)})
    if not rows: raise ValueError('no L90X rows')
    out.mkdir(parents=True,exist_ok=False)
    specs={'image_base':(np.uint8,(224,224,3)),'image_wrist':(np.uint8,(224,224,3)),
        'state':(np.float32,(32,)),'actions':(np.float32,(10,32)),'hist':(np.float32,(10,7)),
        'has_hist':(bool,()),'is_select':(bool,()),'prompt':(np.int32,(200,)),'prompt_mask':(bool,(200,)),
        'task':(np.int32,()),'episode':(np.int32,()),'frame':(np.int32,())}
    for k,(dtype,tail) in specs.items():
        a=np.lib.format.open_memmap(out/f'{k}.npy',mode='w+',dtype=dtype,shape=(len(rows),)+tail)
        for i,r in enumerate(rows):a[i]=r[k]
        a.flush()
    _write_json(out/'manifest.json',{'rows':len(rows),'tasks':[x['task_id'] for x in tasks],
        'fit_tasks':fit_tasks,'selection_tasks':select_tasks,'dev_tasks':list(DEV_TASKS),'eval_tasks':list(EVAL_TASKS),
        'seed':SEED,'demos_per_task':int(demos),'smoke':bool(smoke),'episodes':episodes,
        'windows':'HDF5 every other frame, 10 action frames, stride 10; same pi05 inputs/transforms and image orientation as prepare_oracle',
        'norm_stats':str(norm_path),'norm_sha256':hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        'eval_targets_used_for_split_or_selection':False})
    return {'rows':len(rows),'tasks':len(tasks),'fit_tasks':fit_tasks,'select_tasks':select_tasks}

def extract(out,*,batch=4,threads=8):
    """Extract final prefix tokens, selected mid-layer V, pi05 action and Expert hidden."""
    import torch;torch.set_num_threads(int(threads))
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.25')
    out=Path(out);manifest=json.loads((out/'manifest.json').read_text());n=int(manifest['rows'])
    feat=out/'features';feat.mkdir(parents=True,exist_ok=True)
    if (feat/'complete.json').exists():return json.loads((feat/'complete.json').read_text())
    # No input labels are read here; only observation fields and row indices.
    row={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in out.glob('*.npy')}
    import theta_regression as tr
    base=tr._module(tr.PI05/'run.py','x1_theta_run')
    model,config=base.load_policy(tr.CHECKPOINT)
    import jax,jax.numpy as jnp
    import key_query_probe as kq
    from openpi.shared.nnx_utils import module_jit
    prefix_fn=module_jit(model.prefix_features)
    # Prefix V-cache width from the same Gemma variant; retain only layers 16/17.
    from openpi.models import gemma
    gcfg=gemma.get_config(config.paligemma_variant);vdim=int(gcfg.num_kv_heads*gcfg.head_dim)
    paths={'tokens':feat/'tokens.npy','groups':feat/'groups.npy','mid_v':feat/'mid_v.npy',
           'mid_mask':feat/'mid_mask.npy','pi05':feat/'pi05.npy','expert_t1':feat/'expert_t1.npy'}
    shapes={'tokens':(n,712,2048),'groups':(n,712),'mid_v':(n,2,128+200,vdim),
            'mid_mask':(n,328),'pi05':(n,10,7),'expert_t1':(n,10,int(model.action_out_proj.kernel.value.shape[0]))}
    dtypes={'tokens':np.float16,'groups':np.int8,'mid_v':np.float16,'mid_mask':bool,'pi05':np.float16,'expert_t1':np.float16}
    progress=feat/'progress.json';done=int(json.loads(progress.read_text())['done']) if progress.exists() else 0
    arrays={k:np.lib.format.open_memmap(p,mode='r+' if p.exists() else 'w+',dtype=dtypes[k],shape=shapes[k]) for k,p in paths.items()}
    graph,state=__import__('flax.nnx',fromlist=['split']).split(model)
    # Reusable K1 implementation: prefix V from 16/17, and pi05 action/Expert hidden.
    def tap(st,obs,rids):
        live=__import__('flax.nnx',fromlist=['merge']).merge(graph,st)
        return kq._batched_policy_action_hidden(live,obs,rids,num_steps=10,return_mid_values=True)
    tap=jax.jit(tap)
    for st in range(done,n,int(batch)):
        ids=np.arange(st,min(st+int(batch),n),dtype=np.int64);obs=base.obs_batch(row,ids)
        h,g=prefix_fn(obs);h=np.asarray(h);g=np.asarray(g)
        rid=jnp.asarray(1_000_000+ids,dtype=jnp.uint32)
        act,h1,hf,mv=tap(state,obs,rid)
        # Select only layers 16 and 17 from the K1 extractor's fixed order.
        mv=np.asarray(mv,np.float32)[:,[4,5]]
        # 2x2 mean pool each 16x16 camera; preserve language tokens and mask.
        pooled,mask=kq.pool_prefix_v_layers(mv,np.asarray(row['prompt_mask'][ids],bool),layer_ids=(16,17))
        arrays['tokens'][ids]=h.astype(np.float16);arrays['groups'][ids]=g.astype(np.int8)
        arrays['mid_v'][ids]=pooled.astype(np.float16);arrays['mid_mask'][ids]=mask
        arrays['pi05'][ids]=np.asarray(act,np.float16);arrays['expert_t1'][ids]=np.asarray(h1,np.float16)
        for a in arrays.values():a.flush()
        done=int(ids[-1]+1);_write_json(progress,{'done':done,'total':n,'complete':done==n})
    for a in arrays.values():a.flush()
    outmeta={'complete':True,'rows':n,'layers':[16,17],'final_tokens':[712,2048],
        'mid_v_shape':list(shapes['mid_v'][1:]),'action_noise':'fold_in(key(4101), 1_000_000 + row_id)',
        'target_actions_used':False,'hidden_tap':'K1 suffix hidden at t=1'}
    _write_json(feat/'complete.json',outmeta);return outmeta

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--demos',type=int,default=20);p.add_argument('--batch',type=int,default=4);p.add_argument('--threads',type=int,default=8)
    a=p.parse_args()
    if a.stage=='prepare':print(json.dumps(prepare(a.out,smoke=a.smoke,demos=2 if a.smoke else a.demos),ensure_ascii=False),flush=True)
    else:print(json.dumps(extract(a.out,batch=a.batch,threads=a.threads),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
