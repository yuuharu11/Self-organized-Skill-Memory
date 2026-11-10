"""Unseen-eval layer curves for the K2 per-layer retrievers (models already selected/trained on LIBERO-40).

Scores every prefix / Expert-suffix layer model on unseen LIBERO-90 eval: shape error, gap recovery
(random→oracle), Recall@5, and six-axis + G1-gripper action MSE with θ* (diagnostic) and θ_src.
Unseen targets are used only here, for scoring; nothing is re-selected from these numbers.
"""
import json, sys
from pathlib import Path
import numpy as np
import os
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import key_query_k2 as k2  # noqa: E402
import theta_flow as tf  # noqa: E402

ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
K2P = ROOT / 'key-query-02' / 'production'
OUT = ROOT / 'key-query-02' / 'unseen-layers'


def main(device='cuda'):
    import torch
    import variable_shape_memory as vs
    OUT.mkdir(parents=True, exist_ok=True)
    k1, rows, unseen, splits, data, meta = k2._load_k2_context(K2P, target_actions=True)
    train = json.loads((K2P / 'train_summary.json').read_text())
    theta, pool = k2._k2_pool(k1, train['holdout_tasks'])
    d = data['eval']; n = len(d['ids'])
    z = np.load(ROOT / 'key-query-01' / 'production' / 'evaluation' / 'rows.npz')
    assert np.array_equal(z['row_ids'], d['ids']) and z['shape_error'].shape[1] == len(pool['shape'])
    y = d['actions']; hard = z['hard'].astype(bool); allowed = z['allowed']
    err = np.where(allowed, z['shape_error'], np.inf); orc = err.min(1)
    rnd = np.array([z['shape_error'][i, allowed[i]].mean() for i in range(n)])
    g1 = np.load(tf.OUT / 'eval_rows.npz')  # same row order as K1 (asserted in theta_flow)
    # G1 gripper recomputed from theta_flow results file is per-variant total; rebuild G1 directly
    cond = np.load(tf.OUT / 'eval_cond.npz'); thr = json.loads((tf.OUT / 'results.json').read_text())['gripper_threshold']
    cur = tf.gripper_g1(cond, thr); G1 = ((cur[:, None] - y[:, :, 6]) ** 2).mean(1) / 7
    tphys = vs.to_physical(y[:, :, :6]); state = np.asarray(d['state'], np.float32)
    rowsout = []
    for fam, key in (('prefix', 'tokens'), ('expert', 'suffix')):
        for layer in range(18):
            name = f"{'prefix_l' if fam == 'prefix' else 'expert_v_l'}{layer:02d}_visual"
            ck = K2P / 'models' / f'{name}.pt'
            if not ck.exists():
                continue
            c = torch.load(ck, map_location='cpu', weights_only=False)
            tok = np.asarray(d[key][:, layer], np.float32)
            mask = np.ones(tok.shape[:2], bool) if fam == 'expert' else np.asarray(d['mask'], bool)
            q = k2.make_k2_layer_query(token_dim=tok.shape[-1], history_dim=1, history_mode='visual'); kk = k2.make_candidate_key(70)
            q.load_state_dict(c['query']); kk.load_state_dict(c['key'])
            _, rank = k2.score_layer_retriever({'query': q, 'key': kk}, tok, mask, state, np.zeros((n, 1), np.float32),
                                               pool['shape'], allowed, device=device, batch_size=128)
            pick = rank[:, 0]; se = err[np.arange(n), pick]
            gap = (rnd - se).mean() / (rnd - orc).mean(); gaph = (rnd - se)[hard].mean() / (rnd - orc)[hard].mean()
            eps = .01; pos = err <= orc[:, None] + eps
            r5 = np.mean([pos[i, rank[i, :5]].any() for i in range(n)])
            shp = pool['shape'][pick]
            star = tf.six_axis_mse(tf.decode_many(shp, tf.batched_procrustes(shp[:, :, :6], tphys)[0], vs), y) + G1
            src = tf.six_axis_mse(tf.decode_many(shp, pool['source_theta'][pick], vs), y) + G1
            rec = {'family': fam, 'layer': layer, 'shape': float(se.mean()), 'gap': float(gap), 'gap_hard': float(gaph), 'recall5': float(r5),
                   'theta_star_total': [float(star.mean()), float(star[hard].mean())], 'theta_src_total': [float(src.mean()), float(src[hard].mean())],
                   'holdout_shape': train['models'][name]['val_shape_mse']}
            rowsout.append(rec); print(json.dumps(rec), flush=True)
    (OUT / 'results.json').write_text(json.dumps(rowsout, indent=1))
    L = ['# K2 per-layer retrievers on unseen LIBERO-90 eval (visual-only queries)', '',
         'Selection/training used LIBERO-40 only; these unseen numbers are scoring only.',
         f'random shape {rnd.mean():.4f}, oracle shape {orc.mean():.4f}. Totals = six-axis + G1 gripper.', '',
         '| family | layer | holdout shape | unseen shape | gap all / hard | Recall@5 | total θ* all / hard | total θ_src all / hard |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rowsout:
        L.append(f"| {r['family']} | {r['layer']} | {r['holdout_shape']:.4f} | {r['shape']:.4f} | {100*r['gap']:.1f}% / {100*r['gap_hard']:.1f}% | {r['recall5']:.3f} | "
                 f"{r['theta_star_total'][0]:.3f} / {r['theta_star_total'][1]:.3f} | {r['theta_src_total'][0]:.3f} / {r['theta_src_total'][1]:.3f} |")
    (OUT / 'report.md').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
