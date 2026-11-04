"""Two discarded bridge updates; check expert updates and frozen VLM."""
import argparse
from pathlib import Path
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.shared.nnx_utils import PathRegex
from run import load_policy, load_rows, obs_batch, make_bridge_step, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    model, _ = load_policy(args.checkpoint)
    rows = load_rows(args.output)
    filt = nnx.All(nnx.Param, PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    selected = nnx.state(model, filt)
    count = sum(x.value.size for x in selected.flat_state().values())
    assert count > 100_000_000, count
    def samples():
        return {str(k): np.asarray(v.value).reshape(-1)[::max(1,v.value.size//16)].copy()
                for k,v in nnx.state(model, nnx.All(nnx.Param, nnx.Not(filt))).flat_state().items()}
    frozen = samples()
    before = np.asarray(model.action_out_proj.kernel.value).copy()
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(2.5e-5,b1=.9,b2=.95,weight_decay=1e-5))
    opt = tx.init(selected)
    step = make_bridge_step(filt, tx)
    started = time.monotonic()
    losses = []
    for i in range(2):
        ids = np.arange(8)+8*i
        targets = jnp.asarray(rows['actions'][ids])
        # Plumbing only: rolled targets are NOT retrieved production drafts.
        loss,opt = step(model,opt,obs_batch(rows,ids),targets,jnp.roll(targets,1,axis=1),jax.random.key(100+i))
        losses.append(float(loss))
        print({'step':i+1,'loss':losses[-1],'seconds':time.monotonic()-started},flush=True)
    assert np.isfinite(losses).all()
    change = float(np.max(np.abs(np.asarray(model.action_out_proj.kernel.value)-before)))
    assert change > 0
    for key,value in samples().items():
        np.testing.assert_array_equal(value,frozen[key])
    write_json(args.output/'gradient_smoke.json',{'losses':losses,'trainable_parameters':count,
        'action_output_max_change':change,'sampled_vlm_leaves_unchanged':True,
        'seconds':time.monotonic()-started,'note':'discarded plumbing updates; no production checkpoint'})


if __name__ == '__main__':
    main()
