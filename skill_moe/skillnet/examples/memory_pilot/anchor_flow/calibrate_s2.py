#!/usr/bin/env python3
"""Calibrate retrieval temperature and t0 using unused LIBERO-90 raw episodes.

The inventory is a separately audited match of every fine-tuning/cache episode
to the original HDF5 demos. Only demos outside that exclusion set are accepted.
One demo per task fits the monotone table; another selects tau/confidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import torch

from confidence_calibration import calibrated_t0, denoising_steps, fit_monotone
from memory import (backbone_digest, load_memory, prepare_online_query,
                    retrieval_confidence, sample_memory_anchor)
from openpi import transforms
from openpi.models.model import ModelType
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies.libero_policy import LiberoInputs
from openpi.shared import nnx_utils, normalize
from smoke import DEFAULT_CHECKPOINT, load_model
from train import observation_batch


def heldout_rows(inventory, *, raw90, tasks=32, observations=4):
    if not inventory['all_training_episodes_mapped']:
        raise ValueError('every training episode must first be accounted for in the raw-demo audit')
    available = defaultdict(list)
    for item in inventory['heldout_candidates']:
        task = inventory['tasks'][item['source_file']]
        # Restrict calibration to represented training-suite tasks, not the
        # entirely missing butter task; ensure two genuinely unused episodes.
        if task['training_rows'] and item['length'] >= 30:
            available[item['source_file']].append(item)
    eligible = sorted(k for k, v in available.items() if len(v) >= 2)
    rng = np.random.default_rng(20261028)
    chosen = sorted(rng.choice(eligible, min(tasks, len(eligible)), replace=False))
    annotations = json.loads((Path(__file__).resolve().parents[2] /
                             'libero/annotations/instruct2plan_obj_90.json').read_text())
    norm_file = Path(DEFAULT_CHECKPOINT).parent/'assets/jsw19/libero_90_v1/norm_stats.json'
    processor = transforms.compose([
        LiberoInputs(ModelType.PI05), transforms.Normalize(normalize.load(norm_file.parent), use_quantiles=True),
        transforms.ResizeImages(224,224), transforms.TokenizePrompt(PaligemmaTokenizer(200), discrete_state_input=False),
        transforms.PadStatesAndActions(32)])
    columns = defaultdict(list); selected = []
    for task in chosen:
        demo_ids = rng.choice(len(available[task]), 2, replace=False)
        with h5py.File(Path(raw90)/task) as f:
            language = json.loads(f['data'].attrs['problem_info'])['language_instruction']
            for side, demo_index in zip(('fit', 'select'), demo_ids, strict=True):
                item = available[task][int(demo_index)]
                assert item['demo'] not in inventory['tasks'][task]['excluded_demos']
                demo = f['data'][item['demo']]
                episode = f"heldout-libero90/{task}/{item['demo']}"
                selected.append({**item, 'partition': side, 'episode_id': episode})
                for frame in np.linspace(5, len(demo['actions'])-10, observations, dtype=int):
                    state = np.concatenate([demo['obs/ee_pos'][frame], demo['obs/ee_ori'][frame],
                                            demo['obs/gripper_states'][frame]])
                    row = processor({'observation/state': state,
                        'observation/image': np.ascontiguousarray(demo['obs/agentview_rgb'][frame][::-1,::-1]),
                        'observation/wrist_image': np.ascontiguousarray(demo['obs/eye_in_hand_rgb'][frame][::-1,::-1]),
                        'actions': demo['actions'][frame:frame+10], 'prompt': language,
                        'skills': str(annotations[language]['all_classes'])})
                    for name in ('state','actions','skills','skill_mask','tokenized_prompt','tokenized_prompt_mask'):
                        columns[name].append(row[name])
                    columns['image_base'].append(row['image']['base_0_rgb'])
                    columns['image_left_wrist'].append(row['image']['left_wrist_0_rgb'])
                    for name, value in dict(episode_id=episode, init_state_id=episode, source_file=task,
                                            demo=item['demo'], frame=frame, partition=side).items():
                        columns[name].append(value)
    result = {k: np.asarray(v) for k,v in columns.items()}
    result['suite'] = np.full(len(result['state']), 'libero90')
    result['key_text_source'] = np.asarray('instruction_only')
    result['normalization_sha256'] = np.asarray(hashlib.sha256(norm_file.read_bytes()).hexdigest())
    return result, selected


from research_paths import REPO_ROOT as RESEARCH_REPO_ROOT, STATE_DIR as RESEARCH_STATE_DIR
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--memory', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--raw90', default=str(RESEARCH_REPO_ROOT / 'data/LIBERO/libero_90'))
    parser.add_argument('--tasks', type=int, default=32)
    parser.add_argument('--observations', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--reuse', default='', help='Reuse identical observations/keys and completed paired-error rows from an interrupted calibration')
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if (out/'calibration.json').exists():
        raise FileExistsError(out/'calibration.json')
    started = time.time()
    inventory = json.loads(Path(args.inventory).read_text())
    reuse = Path(args.reuse) if args.reuse else None
    if reuse:
        with np.load(reuse/'heldout-observations.npz', allow_pickle=False) as z:
            data = {k:z[k] for k in z.files}
        selected = json.loads((reuse/'selected-episodes.json').read_text())
        for item in selected:
            assert item['demo'] in inventory['tasks'][item['source_file']]['candidate_demos']
            assert item['demo'] not in inventory['tasks'][item['source_file']]['excluded_demos']
    else:
        data, selected = heldout_rows(inventory, raw90=args.raw90, tasks=args.tasks, observations=args.observations)
    (out/'selected-episodes.json').write_text(json.dumps(selected, indent=2))
    np.savez_compressed(out/'heldout-observations.npz', **data)
    memory = load_memory(args.memory)
    assert not set(data['episode_id']) & {r['meta']['episode_id'] for r in memory.memory}
    model, config = load_model(args.checkpoint)
    encode = nnx_utils.module_jit(model.encode_prefix_key)
    sample = nnx_utils.module_jit(model.sample_actions, static_argnames=('num_steps',))
    size = args.batch_size; n = len(data['state']); keys = []
    assert n % size == 0, 'use complete fixed-size batches to bound JAX compilations'
    if reuse and (reuse/'query-keys.npy').exists():
        keys = np.load(reuse/'query-keys.npy')
    else:
        for lo in range(0,n,size):
            obs = observation_batch({k:v[lo:lo+size] for k,v in data.items() if v.ndim and len(v)==n}, config)
            keys.extend(np.asarray(encode(obs)))
        keys = np.asarray(keys, np.float32)
    np.save(out/'query-keys.npy', keys)
    metadata = [prepare_online_query(memory, key, {'episode_id':str(ep), 'init_state_id':str(ep)})
                for key,ep in zip(keys,data['episode_id'],strict=True)]
    seeds = np.asarray([[7 + i*97 + s*100003 for i in range(n)] for s in range(args.samples)])
    noises = np.asarray([[np.asarray(jax.random.normal(jax.random.split(jax.random.key(int(seed)))[1],(10,32)))
                          for seed in row] for row in seeds])
    grid = np.asarray([.2,.4,.6,.8,1.]); temperatures = [.0001,.001,.01,.07]
    errors_all=[]; stats_all=[]; candidates=[]; reference_anchors=None
    fit = data['partition']=='fit'; select = ~fit
    for tau in temperatures:
        memory.temperature = tau
        stats = np.asarray([retrieval_confidence(memory,key,meta=meta)
                            for key,meta in zip(keys,metadata,strict=True)])
        anchors = np.asarray([[sample_memory_anchor(memory,keys[i],data['state'][i],
                              seed=int(seeds[s,i]) ^ 0x51A2,meta=metadata[i])
                              for i in range(n)] for s in range(args.samples)])
        errs = np.zeros((len(grid),n),np.float64)
        completed_rows = 0
        if reuse and (reuse/f'errors-tau-{tau}.npz').exists():
            with np.load(reuse/f'errors-tau-{tau}.npz') as z:
                assert np.array_equal(z['grid'], grid)
                np.testing.assert_allclose(z['stats'], stats, rtol=0, atol=1e-7)
                completed_rows = int(z['completed_grid_rows'])
                errs[:completed_rows] = z['errors'][:completed_rows]
        # Same anchor + same explicit noise + same observation means identical
        # deterministic sampler output. Reuse only bitwise equal anchor arrays
        # for ALL paired component draws, never approximate nearest matches.
        identical = (np.all(anchors == reference_anchors, axis=(0,2,3))
                     if reference_anchors is not None else np.zeros(n,bool))
        for ti,t0 in enumerate(grid):
            if ti < completed_rows:
                print(json.dumps(dict(stage='reuse_completed_grid',temperature=tau,t0=t0)),flush=True)
                continue
            copied = np.ones(n,bool) if t0==1 and errors_all else identical
            if errors_all:
                errs[ti,copied] = errors_all[0][ti,copied]
            remaining = np.flatnonzero(~copied)
            for s in range(args.samples):
                for lo in range(0,len(remaining),size):
                    ids = remaining[lo:lo+size]; used = len(ids)
                    padded = np.pad(ids,(0,size-used),mode='edge')
                    obs = observation_batch({k:v[padded] for k,v in data.items() if v.ndim and len(v)==n},config)
                    noise = jnp.asarray(noises[s,padded])
                    x = (1.-t0)*jnp.asarray(anchors[s,padded])+t0*noise
                    # Grid aligned to 0.1 makes the Euler step exactly the normal 0.1.
                    result = sample(jax.random.key(17),obs,num_steps=denoising_steps(float(t0)),
                                    x_src=x,t0=1. if t0==1 else jnp.full((size,),float(t0)),noise=noise)
                    err = np.mean((np.asarray(result)[...,:7]-data['actions'][padded,...,:7])**2,axis=(1,2))
                    errs[ti,ids] += err[:used]/args.samples
            print(json.dumps(dict(stage='errors',temperature=tau,t0=t0,num_steps=denoising_steps(float(t0)),
                                  mse=float(errs[ti].mean()),recomputed_rows=len(remaining),
                                  bitwise_identical_reused_rows=int(copied.sum()),elapsed_seconds=time.time()-started)),flush=True)
            # Save completed grid rows to make long calibration reviewable.
            np.savez_compressed(out/f'errors-tau-{tau}.npz', errors=errs, grid=grid,stats=stats,completed_grid_rows=ti+1)
        for metric, confidence in [('max_cosine_similarity',stats[:,0]),('one_minus_normalized_entropy',1.-stats[:,1])]:
            table = fit_monotone(confidence[fit],errs[:,fit],grid,bins=4)
            idx = np.searchsorted(table['edges'],confidence[select],side='right')
            gidx = np.asarray([int(np.argmin(abs(grid-t))) for t in np.asarray(table['t0'])[idx]])
            validation = float(errs[gidx,np.flatnonzero(select)].mean())
            candidates.append(dict(temperature=tau,metric=metric,fit_table=table,selection_mse=validation,
                                   normal_selection_mse=float(errs[-1,select].mean())))
        errors_all.append(errs); stats_all.append(stats)
        if reference_anchors is None:
            reference_anchors=anchors.copy()
        (out/'candidates-progress.json').write_text(json.dumps(candidates,indent=2))
    best = min(candidates,key=lambda c:c['selection_mse'])
    ti = temperatures.index(best['temperature']); stats=stats_all[ti]; errs=errors_all[ti]
    confidence = stats[:,0] if best['metric']=='max_cosine_similarity' else 1.-stats[:,1]
    table = fit_monotone(confidence,errs,grid,bins=4)
    result = dict(**table,metric=best['metric'],temperature=best['temperature'],suite='libero90',
        query_text_source='instruction_only',normalization_sha256=memory.normalization_sha256,
        memory_backbone_sha256=backbone_digest(memory),checkpoint=str(Path(args.checkpoint).resolve()),
        samples=args.samples,rows=n,episodes=len(selected),tasks=len(set(data['source_file'])),
        fit_rows=int(fit.sum()),selection_rows=int(select.sum()),selected_before_refit=best,
        normal_validation_mse=float(errs[-1].mean()),fixed_t0_mse={str(t):float(e.mean()) for t,e in zip(grid,errs,strict=True)},
        error_metric='normalized action chunk MSE, 10 steps x 7 active channels; paired noise/component draws',
        all_training_episodes_mapped=True,raw_demo_exclusion='holdout-action-inventory.json',
        original_pretraining_may_have_seen_LIBERO90=True,
        selection_note='Temperature/confidence chosen using separate raw episodes from table fit. Final table refit on all validation episodes. No LIBERO10/Skill rows. Validation selection scores are not an independent performance claim.',
        elapsed_seconds=time.time()-started,candidates=candidates)
    (out/'calibration.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(out/'calibration-errors.npz',errors=np.asarray(errors_all),stats=np.asarray(stats_all),
                        temperatures=temperatures,t0_grid=grid,partition=data['partition'],episode_id=data['episode_id'])
    times=[calibrated_t0(result,*s)[0] for s in stats]
    (out/'report.md').write_text('# S2 LIBERO-90 保留エピソードによる較正\n\n'+
        f"{len(selected)} episodes / {n} observations / {result['tasks']} tasks; paired samples={args.samples}.\n\n"+
        f"tau={result['temperature']}, confidence={result['metric']}; edges={result['edges']}, t0={result['t0']}.\n\n"+
        f"Selection MSE={best['selection_mse']:.6g}, no-anchor={best['normal_selection_mse']:.6g}. Final refit validation MSE={result['validation_mse']:.6g}, normal={result['normal_validation_mse']:.6g}.\n\n"+
        f"Validation t0 mean/min/max={np.mean(times):.4f}/{min(times):.1f}/{max(times):.1f}.\n\n"+
        'Raw LIBERO-90 demos absent from the fine-tuning/M1/M6 cache were selected using a conservative action-signature exclusion audit. The released original policy may have seen these demonstrations during original pretraining; this is not a pretraining-independent generalization claim. No new-task data were used.\n\n'+
        'Complete temperatures, bin boundaries/counts, unconstrained optima, held-out errors, and episode IDs are in calibration.json, calibration-errors.npz, selected-episodes.json.\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
