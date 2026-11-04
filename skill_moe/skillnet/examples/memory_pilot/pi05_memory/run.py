#!/usr/bin/env python3
"""Resumable LIBERO-40 token-memory experiment, with a frozen dense fallback.

Stages run in separate processes to release accelerator memory between JAX
feature extraction, exact Torch token retrieval, and bridge training.
Large arrays are safe .npy memmaps; no pickle is read.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def log(**values):
    print(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values}), flush=True)


def load_rows(out):
    return {p.stem: np.load(p, mmap_mode='r', allow_pickle=False)
            for p in (out / 'rows').glob('*.npy')}


def calibration_episodes(rows):
    """Stable episode partition, independent of query-window sampling."""
    dev, evaluation = [], []
    for task in np.unique(rows['task']):
        episodes = np.unique(rows['episode'][(rows['task']==task)&rows['validation']])
        order = np.random.default_rng(20261105+int(task)).permutation(episodes)
        dev.extend(order[:3].tolist())
        evaluation.extend(order[3:].tolist())
    if set(dev)&set(evaluation):
        raise ValueError('calibration episode leakage')
    return np.asarray(dev),np.asarray(evaluation)


def obs_batch(rows, ids):
    from openpi.models.model import Observation
    import jax.numpy as jnp
    images = {name: jnp.asarray(rows[field][ids], jnp.float32) / 255 * 2 - 1
              for name, field in [('base_0_rgb', 'image_base'), ('left_wrist_0_rgb', 'image_wrist')]}
    images['right_wrist_0_rgb'] = jnp.zeros_like(images['base_0_rgb'])
    return Observation(
        images=images,
        image_masks={name: jnp.full((len(ids),), name != 'right_wrist_0_rgb') for name in images},
        state=jnp.asarray(rows['state'][ids]),
        tokenized_prompt=jnp.asarray(rows['prompt'][ids]),
        tokenized_prompt_mask=jnp.asarray(rows['prompt_mask'][ids]))


def load_policy(checkpoint):
    from core import make_model
    from openpi.models.model import restore_params
    import flax.nnx as nnx
    model, config = make_model(abstract=True)
    state = nnx.state(model)
    loaded = restore_params(Path(checkpoint) / 'params', restore_type=np.ndarray)
    # Fail rather than silently backfill randomly initialized model leaves.
    expected = state.to_pure_dict()
    import jax
    a, ta = jax.tree.flatten(expected)
    b, tb = jax.tree.flatten(loaded)
    if ta != tb or any(x.shape != y.shape for x, y in zip(a, b, strict=True)):
        raise ValueError('dense LIBERO checkpoint does not exactly match model shape/tree')
    loaded = jax.tree.map(jax.numpy.asarray, loaded)
    state.replace_by_pure_dict(loaded)
    nnx.update(model, state)
    return model, config


def prepare(args):
    import pyarrow.parquet as pq
    from PIL import Image
    from openpi.transforms import Normalize, ResizeImages, TokenizePrompt, PadStatesAndActions
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.models.model import ModelType
    from openpi.shared.normalize import load

    out, source = args.output, args.dataset
    norm_files = list((args.checkpoint / 'assets').rglob('norm_stats.json'))
    if len(norm_files) != 1:
        raise ValueError(f'expected one LIBERO normalization file, got {norm_files}')
    norm_file = norm_files[0]
    tasks = {int(x['task_index']): x['task'] for x in
             map(json.loads, (source / 'meta/tasks.jsonl').read_text().splitlines())}
    if len(tasks) != 40:
        raise ValueError('expected exactly the LIBERO-40 task inventory')
    files = sorted((source / 'data').rglob('*.parquet'))
    records, by_task = [], {}
    for p in files:
        t = pq.read_table(p, columns=['task_index', 'episode_index'])
        ts, es = np.unique(t['task_index'].to_numpy()), np.unique(t['episode_index'].to_numpy())
        if len(ts) != 1 or len(es) != 1:
            raise ValueError(f'episode file contains multiple tasks/episodes: {p}')
        rec = {'path': str(p), 'task': int(ts[0]), 'episode': int(es[0]), 'length': len(t)}
        records.append(rec)
        by_task.setdefault(rec['task'], []).append(rec)
    rng = np.random.default_rng(args.seed)
    for task, group in sorted(by_task.items()):
        order = rng.permutation(len(group))
        if len(group) < 15:
            raise ValueError(f'task {task} has too few episodes')
        for i, j in enumerate(order):
            group[j]['split'] = 'validation' if i < 5 else 'train'
            group[j]['memory'] = 5 <= i < 15
    row_records = [(rec, frame) for rec in records
                   for frame in range(0, rec['length'] - 9, 10)]
    n = len(row_records)
    dest = out / 'rows'
    dest.mkdir(exist_ok=True)
    specs = {'image_base': ((n,224,224,3), 'uint8'), 'image_wrist': ((n,224,224,3), 'uint8'),
             'state': ((n,32), 'float32'), 'actions': ((n,10,32), 'float32'),
             'prompt': ((n,200), 'int32'), 'prompt_mask': ((n,200), 'bool'),
             'episode': ((n,), 'int32'), 'task': ((n,), 'int32'), 'frame': ((n,), 'int32'),
             'validation': ((n,), 'bool'), 'memory': ((n,), 'bool')}
    arrays = {k: np.lib.format.open_memmap(dest / (k+'.npy'), mode='w+', shape=s, dtype=d)
              for k,(s,d) in specs.items()}
    transforms = [LiberoInputs(ModelType.PI05), Normalize(load(norm_file.parent), use_quantiles=True),
                  ResizeImages(224,224), TokenizePrompt(PaligemmaTokenizer(200), discrete_state_input=False),
                  PadStatesAndActions(32)]
    index = 0
    for ei, rec in enumerate(records):
        t = pq.read_table(rec['path'], columns=['image','wrist_image','state','actions'])
        actions = np.asarray(t['actions'].to_pylist(), np.float32)
        states = np.asarray(t['state'].to_pylist(), np.float32)
        for frame in range(0, rec['length'] - 9, 10):
            images = {}
            for col in ['image','wrist_image']:
                obj = t[col][frame].as_py()
                with Image.open(io.BytesIO(obj['bytes'])) as im:
                    images[col] = np.asarray(im.convert('RGB'))
            item = {'observation/image': images['image'], 'observation/wrist_image': images['wrist_image'],
                    'observation/state': states[frame], 'actions': actions[frame:frame+10],
                    'prompt': tasks[rec['task']]}
            for transform in transforms:
                item = transform(item)
            arrays['image_base'][index] = item['image']['base_0_rgb']
            arrays['image_wrist'][index] = item['image']['left_wrist_0_rgb']
            for key in ['state','actions']:
                arrays[key][index] = item[key]
            arrays['prompt'][index] = item['tokenized_prompt']
            arrays['prompt_mask'][index] = item['tokenized_prompt_mask']
            for key, value in [('episode',rec['episode']), ('task',rec['task']), ('frame',frame),
                               ('validation',rec['split']=='validation'), ('memory',rec['memory'])]:
                arrays[key][index] = value
            index += 1
        if ei % 50 == 0:
            log(stage='prepare', episodes=ei, rows=index, total=n)
    for a in arrays.values():
        a.flush()
    write_json(out / 'data_manifest.json', {
        'source': str(source), 'source_info': json.loads((source/'meta/info.json').read_text()),
        'checkpoint': str(args.checkpoint), 'norm_sha256': hashlib.sha256(norm_file.read_bytes()).hexdigest(),
        'rows': n, 'horizon':10, 'stride':10, 'seed':args.seed, 'tasks':tasks, 'episodes':records,
        'validation_note':'Held out from new codec/bridge fitting; public base policy may have seen these demonstrations.',
        'control_fps':json.loads((source/'meta/info.json').read_text())['fps']})


def extract(args):
    import flax.nnx as nnx
    import jax
    from openpi.shared.nnx_utils import module_jit
    rows = load_rows(args.output)
    model, _ = load_policy(args.checkpoint)
    encode = module_jit(model.prefix_features)
    n, batch = len(rows['actions']), args.feature_batch
    resume_path = args.output/'extraction_resume.json'
    resume = json.loads(resume_path.read_text()) if resume_path.exists() else None
    offset = int(resume['resume_from_row']) if resume else 0
    if not 0 <= offset <= n:
        raise ValueError('invalid extraction resume offset')
    mode = 'r+' if resume else 'w+'
    tokens = np.lib.format.open_memmap(args.output/'tokens.npy', mode=mode, dtype='float16', shape=(n,712,2048))
    groups = np.lib.format.open_memmap(args.output/'groups.npy', mode=mode, dtype='int8', shape=(n,712))
    pooled = np.lib.format.open_memmap(args.output/'pooled.npy', mode=mode, dtype='float32', shape=(n,2048))
    if tokens.shape != (n,712,2048) or groups.shape != (n,712) or pooled.shape != (n,2048):
        raise ValueError('resume arrays do not match the extraction contract')
    started = time.monotonic()
    for start in range(offset,n,batch):
        ids = np.minimum(np.arange(start,start+batch), n-1)
        hidden, group = encode(obs_batch(rows, ids))
        h, g = np.asarray(hidden), np.asarray(group)
        count = min(batch,n-start)
        tokens[start:start+count] = h[:count]
        groups[start:start+count] = g[:count]
        valid = g[:count] >= 0
        pooled[start:start+count] = (h[:count]*valid[...,None]).sum(1)/valid.sum(1)[:,None]
        if (start-offset) % (batch*20) == 0:
            for array in [tokens,groups,pooled]:
                array.flush()
            write_json(resume_path,{'resume_from_row':start+count,'total_rows':n})
            log(stage='extract', rows=start+count, total=n, elapsed=time.monotonic()-started)
    for a in [tokens,groups,pooled]:
        a.flush()
    write_json(args.output/'features.json', {'rows':n,'tokens':712,'width':2048,
               'groups':['base_camera','wrist_camera','instruction'], 'seconds':time.monotonic()-started,
               'checkpoint':str(args.checkpoint), 'parameters':str(nnx.state(model).to_pure_dict().keys())})


def codec(args):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(HERE.parents[1]))
    from emergent.light.methods.m1 import M1
    rows = load_rows(args.output)
    features = np.load(args.output/'pooled.npy', mmap_mode='r')
    train = np.flatnonzero(~rows['validation'])
    # M1's internal key calibration uses a bounded diagnostic subset; the keys
    # here are for that bookkeeping only, never the token retrieval index.
    model = M1(epochs=30, batch_size=128, seed_value=args.seed, base_dim=12, residual_dim=4)
    model.fit({'actions':np.asarray(rows['actions'][train]), 'state':np.asarray(rows['state'][train]),
               'feature':np.asarray(features[train]), 'key':np.asarray(features[train]),
               'episode_id':np.asarray(rows['episode'][train]).astype(str),
               'init_state_id':np.asarray(rows['episode'][train]).astype(str), 'seed':args.seed})
    payload = {'stats':{k:torch.tensor(getattr(model,k)) for k in ['am','ass','sm','ss','fm','fs']},
               'nets':{k:getattr(model,k).state_dict() for k in ['encoder','prior_net','base_decoder','residual_decoder']}}
    torch.save(payload,args.output/'codec.pt')
    base, residual = model.encode(np.asarray(rows['actions']),np.asarray(rows['state']))
    _, predicted_r = model.prior(np.asarray(features),np.asarray(rows['state']))
    own = model.decode(base,residual,np.asarray(features),np.asarray(rows['state']))
    predicted = model.decode(base,predicted_r,np.asarray(features),np.asarray(rows['state']))
    np.save(args.output/'bases.npy',base)
    np.save(args.output/'predicted_residual.npy',predicted_r)
    np.savez(args.output/'codec_errors.npz', own=np.mean((own[:,:,:7]-rows['actions'][:,:,:7])**2,axis=(1,2)),
             predicted_r=np.mean((predicted[:,:,:7]-rows['actions'][:,:,:7])**2,axis=(1,2)),
             episode=rows['episode'],task=rows['task'],validation=rows['validation'])
    write_json(args.output/'codec.json', {'base_dim':12,'residual_dim':4,'epochs':model.epochs_run,
        'validation_reconstruction_mse':float(np.mean((own[rows['validation'],:,:7]-rows['actions'][rows['validation'],:,:7])**2)),
        'validation_predicted_r_mse':float(np.mean((predicted[rows['validation'],:,:7]-rows['actions'][rows['validation'],:,:7])**2))})


def restore_codec(out):
    import torch
    from torch import nn
    sys.path.insert(0,str(HERE.parents[1]))
    from emergent.light.methods.m1 import M1
    from emergent.light.methods._common import AdditiveActionDecoder
    model = M1(base_dim=12,residual_dim=4)
    model.action_shape=(10,32)
    model.a_dim,model.state_dim,model.feature_dim=320,32,2048
    payload = torch.load(out/'codec.pt',map_location='cpu',weights_only=True)
    for k,v in payload['stats'].items():
        setattr(model,k,v.numpy())
    for name, input_dim, output_dim in [('encoder',352,20),('prior_net',2080,16),
                                        ('base_decoder',44,320),('residual_decoder',4,320)]:
        net=nn.Sequential(nn.Linear(input_dim,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,output_dim))
        net.load_state_dict(payload['nets'][name]);net.eval();setattr(model,name,net)
    return model


def retrieve(args):
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('exact token search requires GPU')
    rows=load_rows(args.output)
    tokens=np.load(args.output/'tokens.npy',mmap_mode='r')
    groups=np.load(args.output/'groups.npy',mmap_mode='r')
    memory_ids=np.flatnonzero(rows['memory'])
    np.save(args.output/'memory_row_ids.npy',memory_ids)
    # Keep each normalized token; resident keys avoid repeatedly copying the bank.
    bank=torch.as_tensor(np.asarray(tokens[memory_ids]),device='cuda',dtype=torch.float16)
    bank_groups=torch.as_tensor(np.asarray(groups[memory_ids]),device='cuda')
    bank=bank/torch.linalg.vector_norm(bank.float(),dim=-1,keepdim=True).clamp_min(1e-8).half()
    episodes=np.asarray(rows['episode'][memory_ids])
    top_ids=np.lib.format.open_memmap(args.output/'retrieved_row.npy',mode='w+',dtype='int64',shape=(len(tokens),))
    top_scores=np.lib.format.open_memmap(args.output/'retrieved_score.npy',mode='w+',dtype='float32',shape=(len(tokens),))
    started=time.monotonic()
    # Token positions are fixed camera blocks followed by text; text validity
    # can contain holes (special tokens), so keep through the last valid index.
    text_columns = np.flatnonzero(np.any(np.asarray(groups[memory_ids])[:,512:] == 2, axis=0))
    text_stop = 512 + int(text_columns[-1]) + 1
    with torch.inference_mode():
        for i in range(len(tokens)):
            query=torch.as_tensor(np.asarray(tokens[i]).copy(),device='cuda').float()
            query=(query/torch.linalg.vector_norm(query,dim=-1,keepdim=True).clamp_min(1e-8)).half()
            qgroups=np.asarray(groups[i])
            scores=torch.empty(len(memory_ids),device='cuda',dtype=torch.float32)
            for start in range(0,len(memory_ids),args.search_block):
                end=min(start+args.search_block,len(memory_ids))
                parts=[]
                for group in [0,1,2]:
                    q=query[qgroups==group]
                    left,right = [(0,256),(256,512),(512,text_stop)][group]
                    keys=bank[start:end,left:right]
                    sim=torch.matmul(q[None],keys.transpose(1,2)).float()
                    sim.masked_fill_(bank_groups[start:end,None,left:right]!=group,-torch.inf)
                    parts.append(sim.amax(-1).mean(-1))
                scores[start:end]=torch.stack(parts).mean(0)
            scores[torch.as_tensor(episodes==rows['episode'][i],device='cuda')]=-torch.inf
            best=int(scores.argmax().item())
            if not torch.isfinite(scores[best]):
                raise ValueError('no cross-episode search candidate')
            top_ids[i]=memory_ids[best]
            top_scores[i]=float(scores[best])
            if i%20==0:
                top_ids.flush();top_scores.flush()
                log(stage='retrieve',rows=i+1,total=len(tokens),memory_rows=len(memory_ids),
                    elapsed=time.monotonic()-started,seconds_per_query=(time.monotonic()-started)/(i+1))
    del bank,bank_groups
    torch.cuda.empty_cache()
    model=restore_codec(args.output)
    bases=np.load(args.output/'bases.npy',mmap_mode='r')
    features=np.load(args.output/'pooled.npy',mmap_mode='r')
    residual=np.load(args.output/'predicted_residual.npy',mmap_mode='r')
    drafts=model.decode(np.asarray(bases[top_ids]),np.asarray(residual),np.asarray(features),np.asarray(rows['state']))
    np.save(args.output/'drafts.npy',drafts)
    write_json(args.output/'retrieval.json',{'rows':len(tokens),'memory_rows':len(memory_ids),
        'seconds':time.monotonic()-started,'top_k':1,'mode':'exact grouped token MaxSim',
        'same_episode_matches':int(np.sum(rows['episode'][top_ids]==rows['episode'])),
        'validation_draft_mse':float(np.mean((drafts[rows['validation'],:,:7]-rows['actions'][rows['validation'],:,:7])**2))})


def make_bridge_step(filt, tx):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    @nnx.jit
    def step(model,opt,obs,actions,draft,rng):
        rng1,rng2,rng3=jax.random.split(rng,3)
        t0=jax.random.uniform(rng1,(actions.shape[0],),minval=.2,maxval=.6)
        progress=jax.random.uniform(rng2,(actions.shape[0],),minval=.01,maxval=1.)
        eps=jax.random.normal(rng3,actions.shape)
        start=t0[:,None,None]
        source=(1-start)*draft+start*eps
        x=(1-progress[:,None,None])*actions+progress[:,None,None]*source
        target=(source-actions)/start
        def loss_fn(current):
            v=current.velocity(obs,x,t0*progress)
            return jnp.mean(jnp.square(v[:,:,:7]-target[:,:,:7]))
        loss,grads=nnx.value_and_grad(loss_fn,argnums=nnx.DiffState(0,filt))(model)
        params=nnx.state(model,filt)
        updates,opt=tx.update(grads,opt,params)
        nnx.update(model,optax.apply_updates(params,updates))
        return loss,opt
    return step


def train(args):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex
    rows=load_rows(args.output)
    drafts=np.load(args.output/'drafts.npy',mmap_mode='r')
    model,_=load_policy(args.checkpoint)
    filt=nnx.All(nnx.Param,PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    selected=nnx.state(model,filt)
    if sum(x.value.size for x in selected.flat_state().values())<100_000_000:
        raise ValueError('action expert filter did not select the dense tower')
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(2.5e-5,b1=.9,b2=.95,weight_decay=1e-5))
    opt=tx.init(selected)
    step=make_bridge_step(filt,tx)
    @nnx.jit
    def validation_step(current,obs,actions,draft,noise):
        start=.4
        source=(1-start)*draft+start*noise
        x=.5*actions+.5*source
        v=current.velocity(obs,x,jnp.full((len(actions),),.2))
        return jnp.mean(jnp.square(v[:,:,:7]-((source-actions)/start)[:,:,:7]))
    dev_episodes,_=calibration_episodes(rows)
    validation_ids=[]
    for task in np.unique(rows['task']):
        available=np.flatnonzero((rows['task']==task)&np.isin(rows['episode'],dev_episodes))
        validation_ids.append(int(available[len(available)//2]))
    validation_ids=np.asarray(validation_ids)
    best_validation=float('inf')
    best_checkpoint=None
    train_ids=np.flatnonzero(~rows['validation'])
    rng=np.random.default_rng(args.seed)
    key=jax.random.key(args.seed)
    logs=[]
    started=time.monotonic()
    for i in range(args.steps):
        ids=rng.choice(train_ids,args.train_batch,replace=False)
        key,k=jax.random.split(key)
        loss,opt=step(model,opt,obs_batch(rows,ids),jnp.asarray(rows['actions'][ids]),jnp.asarray(drafts[ids]),k)
        value=float(loss)
        if not np.isfinite(value):
            raise FloatingPointError('nonfinite bridge loss')
        if i%20==0 or i+1==args.steps:
            record={'stage':'train','step':i+1,'loss':value,'elapsed':time.monotonic()-started}
            logs.append(record);log(**record)
            write_json(args.output/'training_progress.json',{'status':'running','logs':logs,'target_steps':args.steps})
        if (i+1)%500==0 or i+1==args.steps:
            checkpoint=args.output/'bridge'/f'{i+1:06d}'
            checkpoint.parent.mkdir(exist_ok=True)
            ocp.PyTreeCheckpointer().save(checkpoint,nnx.state(model,filt).to_pure_dict())
            write_json(args.output/'latest_bridge.json',{'path':str(checkpoint),'steps':i+1,
                'format':'partial dense action-expert weights; overlay onto manifest fallback checkpoint'})
            values=[]
            for vi in range(0,len(validation_ids),args.train_batch):
                valid_ids=validation_ids[vi:vi+args.train_batch]
                eps=jnp.asarray(np.stack([np.asarray(jax.random.normal(
                    jax.random.fold_in(jax.random.key(77123),int(row)),(10,32))) for row in valid_ids]))
                values.append(float(validation_step(model,obs_batch(rows,valid_ids),
                    jnp.asarray(rows['actions'][valid_ids]),jnp.asarray(drafts[valid_ids]),eps)))
            validation_loss=float(np.mean(values))
            if not np.isfinite(validation_loss):
                raise FloatingPointError('nonfinite held-out bridge loss')
            if validation_loss<best_validation:
                best_validation=validation_loss
                best_checkpoint={'path':str(checkpoint),'steps':i+1,'validation_velocity_mse':validation_loss,
                    'selection':'40 task-balanced dev-episode windows; calibration Eval episodes untouched'}
                write_json(args.output/'best_bridge.json',best_checkpoint)
            log(stage='validation',step=i+1,velocity_mse=validation_loss,best=best_validation)
        if time.monotonic()-started>args.train_hours*3600:
            checkpoint=args.output/'bridge'/f'final-{i+1:06d}'
            checkpoint.parent.mkdir(exist_ok=True)
            ocp.PyTreeCheckpointer().save(checkpoint,nnx.state(model,filt).to_pure_dict())
            write_json(args.output/'latest_bridge.json',{'path':str(checkpoint),'steps':i+1,'budget_stopped':True})
            break
    write_json(args.output/'trained.json',{'steps':i+1,'seconds':time.monotonic()-started,'logs':logs,
        'start_noise_range':[.2,.6],'nfe':10,'loss':'bridge velocity active 7',
        'fallback_checkpoint':str(args.checkpoint),'vlm_frozen':True,'best_checkpoint':best_checkpoint})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['pipeline','prepare','extract','codec','retrieve','train'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=REPO/'.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')
    parser.add_argument('--dataset',type=Path,default=REPO/'.skillnet-state/data/lerobot/emergent/libero_40_unsegmented')
    parser.add_argument('--seed',type=int,default=20261104)
    parser.add_argument('--feature-batch',type=int,default=4)
    parser.add_argument('--search-block',type=int,default=32)
    parser.add_argument('--train-batch',type=int,default=8)
    parser.add_argument('--steps',type=int,default=4000)
    parser.add_argument('--train-hours',type=float,default=4.)
    args=parser.parse_args()
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    if args.stage!='pipeline':
        globals()[args.stage](args);return
    stages=[('prepare','data_manifest.json'),('extract','features.json'),('codec','codec.json'),
            ('retrieve','retrieval.json'),('train','trained.json')]
    write_json(args.output/'experiment.json',{'purpose':'LIBERO-40 pretrained pi0.5 + token motion memory + separate bridge expert',
        'checkpoint':str(args.checkpoint),'dataset':str(args.dataset),'seed':args.seed,
        'references':{'official_pi05_long':92.4,'skillnet40_long':98.4},
        'reference_note':'Published cross-paper values only; no matched reproduction claimed.',
        'scope':'training/preparation; closed-loop success is not yet measured',
        'token_keys':'all valid final VLM tokens, modality-matched MaxSim, no learned retrieval projection',
        'noise':[.2,.6],'fallback':'unchanged official pi05_libero',
        'threshold':'not selected; must calibrate after training without evaluation rollouts'})
    started=time.monotonic()
    for stage,marker in stages:
        if (args.output/marker).exists():
            continue
        write_json(args.output/'status.json',{'status':'running','stage':stage,'pid':os.getpid(),
                    'elapsed_seconds':time.monotonic()-started})
        command=[sys.executable,str(Path(__file__).resolve()),stage,*sys.argv[2:]]
        with (args.output/(stage+'.log')).open('a') as stream:
            proc=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if proc.returncode:
            write_json(args.output/'status.json',{'status':'failed','stage':stage,'returncode':proc.returncode})
            raise SystemExit(proc.returncode)
        if not (args.output/marker).exists():
            raise RuntimeError(f'{stage} did not produce its completion marker')
    if not (args.output/'calibration.json').exists():
        write_json(args.output/'status.json',{'status':'running','stage':'calibrate','pid':os.getpid()})
        with (args.output/'calibrate.log').open('a') as stream:
            proc=subprocess.run([sys.executable,str(HERE/'calibrate.py'),'--output',str(args.output),
                                 '--checkpoint',str(args.checkpoint)],stdout=stream,stderr=subprocess.STDOUT)
        if proc.returncode:
            write_json(args.output/'status.json',{'status':'failed','stage':'calibrate','returncode':proc.returncode})
            raise SystemExit(proc.returncode)
    write_json(args.output/'status.json',{'status':'training_complete','elapsed_seconds':time.monotonic()-started,
        'next':'matched closed-loop evaluation; no success-rate claim yet'})


if __name__=='__main__':
    main()
