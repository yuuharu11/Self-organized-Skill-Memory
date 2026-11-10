"""Apply the Dev-selected L-vlm LoRA theta head to K1-selected pool shapes (unseen eval).
Unseen targets are used only for scoring. Output: per-method 6-axis MSE with LoRA theta."""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = str(Path(__file__).resolve().parent)
sys.path.insert(0, HERE)
import theta_regression as tr  # noqa: E402
import theta_regression_lora as L  # noqa: E402
import variable_shape_memory as vs  # noqa: E402

STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()

E = str(STATE_DIR / 'outputs/memory-pilot/unseen-libero/key-query-01/production/') + '/'
L_OUT = str(STATE_DIR / 'outputs/memory-pilot/unseen-libero/theta-regression-01/lora/production')
OUTF = STATE_DIR / 'outputs/memory-pilot/unseen-libero/theta-flow-01/lora_theta_selected.npz'


def main():
    import flax.nnx as nnx, jax, jax.numpy as jnp
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import module_jit
    L.OUT = Path(L_OUT)
    arch, rep = 'L-vlm', 'C_vm_none'
    spec = L.select_exp1_spec()['selected']
    fit_dir = L._condition_dir(arch, rep)
    best = json.loads((fit_dir / 'best.json').read_text())
    model = L._make_model(arch); L._restore_dense_checkpoint(model)
    W = L._wrapper_type(); wrapper = W(model, arch, 3 * 2048 + 32 + 23, spec['cam_mode'])
    state = nnx.state(wrapper, L._trainable_filter())
    saved = ocp.PyTreeCheckpointer().restore(best['path'])
    state.replace_by_pure_dict(jax.tree.map(jnp.asarray, saved)); nnx.update(wrapper, state)
    trained = json.loads((fit_dir / 'trained.json').read_text())
    mu = np.asarray(trained['theta_mean'], np.float32); sd = np.asarray(trained['theta_std'], np.float32)
    base_mod, rows = L._load_rows_for_lora(); unseen = L._load_unseen()
    z = np.load(E + 'evaluation/rows.npz'); s = json.load(open(E + 'evaluation/summary.json'))
    p0 = tr.candidate_shape_pool(rep=rep); keep = ~np.isin(p0['source_task'], s['holdout_tasks'])
    C = p0['shape'][keep]; TS = p0['source_theta'][keep]
    ids = np.asarray(z['row_ids'], np.int64)  # unseen local row ids
    err = np.where(z['allowed'], z['shape_error'], np.inf)
    picks = {'oracle_shape': err.argmin(1)}
    for k in ['P5_hist', 'P6_tok_hist', 'P1_tok', 'P2_mid_l14', 'P0_mean']:
        picks[k] = z['rank_' + k][:, 0]
    fn = module_jit(wrapper.__call__)
    y = z['target'][:, :, :7]
    out = {}
    for name, pick in picks.items():
        preds = []
        for st in range(0, len(ids), 8):
            sl = slice(st, min(st + 8, len(ids)))
            obs = base_mod.obs_batch(unseen, ids[sl])
            sh = jnp.asarray(C[pick[sl]]); ui = jnp.zeros((sh.shape[0], 10), jnp.int32)
            src = jnp.asarray(np.repeat(TS[pick[sl]][:, None, :], 10, axis=1))
            pn = np.asarray(fn(obs, sh, ui, src))
            preds.append(L._decode_rows(sh, pn, mu, sd, vs))
        a = np.concatenate(preds)
        out[name] = a
        six = ((a[:, :, :6] - y[:, :, :6]) ** 2).mean((1, 2)) * 6 / 7
        print(name, 'six-axis LoRA-theta %.4f' % six.mean(), flush=True)
    OUTF.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTF, **out)


if __name__ == '__main__':
    main()
