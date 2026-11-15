"""PI0REF: ordinary public pi0.5, CMP3-identical rows, active7 MSE.

Reuses corrected official-noise Eval predictions; generates only missing seen
predictions plus an eight-row fresh reproduction check. All outputs are local
to the new experiment. No fitting, retrieval, clipping, or gripper thresholding.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'pi05_memory')]
STATE=Path(os.environ.get('SKILLNET_STATE_DIR','/work/.skillnet-state'))
ROOT=STATE/'outputs/memory-pilot/unseen-libero'
OUT=ROOT/'pi05-reference-01'
RUN=STATE/'outputs/memory-pilot/pi05-memory40-20261104/run-01'
UNSEEN=ROOT/'unseen-rows-02'
CHECKPOINT=STATE/'checkpoints/openpi-assets/checkpoints/pi05_libero'
SEED=4101


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def save_json(path,x):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n')
    tmp.replace(path)


def save_npz(path,**arrays):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(tmp,**arrays);tmp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log(out,stage,**more):
    x={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'stage':stage,**more}
    line=json.dumps(x,ensure_ascii=False);print(line,flush=True)
    with (out/'progress.log').open('a') as f:f.write(line+'\n')


def metric_rows(pred,target):
    pred=np.asarray(pred,np.float64)[:,:10,:7]
    target=np.asarray(target,np.float64)[:,:10,:7]
    if pred.shape!=target.shape or pred.shape[1:]!=(10,7):raise ValueError((pred.shape,target.shape))
    if not np.isfinite(pred).all() or not np.isfinite(target).all():raise ValueError('nonfinite action')
    squared=(pred-target)**2
    arm=squared[:,:,:6].sum((1,2))/70.
    grip=squared[:,:,6].sum(1)/70.
    total=squared.mean((1,2))
    np.testing.assert_allclose(arm+grip,total,rtol=1e-14,atol=1e-14)
    return {'active7':total,'arm_contribution':arm,'gripper_contribution':grip}


def summarize(pred,target,hard):
    rows=metric_rows(pred,target);hard=np.asarray(hard,bool)
    return {name:{'rows':int(mask.sum()),**{k:float(v[mask].mean()) for k,v in rows.items()}}
            for name,mask in [('all',np.ones(len(hard),bool)),('hard',hard)]}


def data_rows(path):
    return {name:np.load(Path(path)/f'{name}.npy',mmap_mode='r',allow_pickle=False)
            for name in ('image_base','image_wrist','state','prompt','prompt_mask','actions','task','episode','frame')}


def preflight(out):
    out.mkdir(parents=True,exist_ok=True);(out/'predictions').mkdir(exist_ok=True)
    if (out/'selection.npz').exists():
        log(out,'preflight_already_complete');return
    log(out,'preflight_start')
    selection=json.loads((ROOT/'compress-seen-cuts-01/production-control/evaluation-selection.json').read_text())
    seen_ids=np.asarray(selection['seen_row_ids'],np.int64)
    rows=data_rows(RUN/'rows'); unseen=data_rows(UNSEEN)
    val=np.load(RUN/'rows/validation.npy',mmap_mode='r');mem=np.load(RUN/'rows/memory.npy',mmap_mode='r')
    assert len(seen_ids)==621 and val[seen_ids].all() and not mem[seen_ids].any()
    demos=set(zip(rows['task'][seen_ids].tolist(),rows['episode'][seen_ids].tolist()))
    assert demos==set(map(tuple,selection['seen_demo_pairs'])) and len(demos)==40
    assert len(np.unique(rows['task'][seen_ids]))==20
    assert not demos & set(zip(rows['task'][mem].tolist(),rows['episode'][mem].tolist()))
    # Exactly the CMP3 hard definition and float32 nearest-template arithmetic.
    target_seen=np.asarray(rows['actions'][seen_ids,:,:7],np.float32)
    memory=np.asarray(rows['actions'][mem,:,:7],np.float32)
    best=np.array([np.mean((memory-y)**2,axis=(1,2)).min() for y in target_seen])
    seen_threshold=float(np.quantile(best,.8));seen_hard=best>=seen_threshold
    assert seen_hard.sum()==125
    eval_cache=load(ROOT/'orientation-fix-01/pi05/eval_global.npz')
    uids=np.asarray(eval_cache['row'],np.int64)
    expected=np.flatnonzero(~np.load(UNSEEN/'is_dev.npy'))
    np.testing.assert_array_equal(uids,expected)
    inp=load(ROOT/'scale-only-02/production/prep/W_inputs_eval.npz')
    np.testing.assert_array_equal(inp['row'],uids)
    for k in ('task','episode','frame'):np.testing.assert_array_equal(inp[k],unseen[k][uids])
    unseen_hard=np.asarray(inp['hard'],bool)
    raw=load(ROOT/'oracle-01/drafts/raw_none_eval.npz')
    np.testing.assert_array_equal(raw['query_ids'],uids+len(rows['task']))
    hard_threshold=json.loads((ROOT/'oracle-01/hard_threshold.json').read_text())['threshold']
    np.testing.assert_array_equal(unseen_hard,raw['pre_mse']>=hard_threshold)
    assert len(uids)==619 and unseen_hard.sum()==134
    norm=CHECKPOINT/'assets/physical-intelligence/libero/norm_stats.json'
    norm_sha=sha(norm)
    assert json.loads((UNSEEN/'manifest.json').read_text())['norm_sha256']==norm_sha
    assert json.loads((RUN/'data_manifest.json').read_text())['norm_sha256']==norm_sha
    target=np.asarray(unseen['actions'][uids,:,:7],np.float32)
    primary=summarize(eval_cache['actions'],target,unseen_hard)
    assert abs(primary['all']['active7']-.1413)<.00005
    assert abs(primary['hard']['active7']-.2870)<.0001
    controls={}
    sources={'corrected_global':(ROOT/'orientation-fix-01/pi05/eval_global.npz','actions'),
        'corrected_k1':(ROOT/'orientation-fix-01/pi05/eval_k1.npz','actions'),
        'legacy_k1':(ROOT/'orientation-fix-01/pi05/legacy_eval_k1.npz','actions'),
        'LIB1_fast_asis':(ROOT/'fast-query-01/production/predictions/pi05_eval_asis.npz','actions'),
        'fast_rot180':(ROOT/'fast-query-01/production/predictions/pi05_eval_rot180.npz','actions'),
        'legacy_global':(ROOT/'oracle-01/official_normal_eval.npz','prediction')}
    predictions={}
    for name,(path,key) in sources.items():
        z=load(path)
        if 'row' in z:np.testing.assert_array_equal(z['row'],uids)
        else:np.testing.assert_array_equal(z['row_ids'],uids+len(rows['task']))
        predictions[name]=z[key]
        controls[name]={'source':str(path),'sha256':sha(path),'metrics':summarize(z[key],target,unseen_hard)}
    assert np.array_equal(predictions['legacy_k1'],predictions['LIB1_fast_asis'])
    assert np.array_equal(predictions['corrected_k1'],predictions['fast_rot180'])
    np.testing.assert_array_equal(inp['pi05'],predictions['corrected_k1'])
    legacy=data_rows(ROOT/'oracle-01/unseen_rows')
    for k in ('state','prompt','prompt_mask','actions','task','episode','frame'):
        np.testing.assert_array_equal(unseen[k][uids],legacy[k][uids])
    for k in ('image_base','image_wrist'):
        np.testing.assert_array_equal(unseen[k][uids],legacy[k][uids][:,::-1,::-1,:])
    save_npz(out/'selection.npz',learned_row=seen_ids,learned_hard=seen_hard,unseen_row=uids,unseen_hard=unseen_hard,
       learned_task=rows['task'][seen_ids],learned_episode=rows['episode'][seen_ids],learned_frame=rows['frame'][seen_ids],
       unseen_task=unseen['task'][uids],unseen_episode=unseen['episode'][uids],unseen_frame=unseen['frame'][uids])
    save_npz(out/'predictions/unseen_cached.npz',actions=eval_cache['actions'],row=uids,target=target,hard=unseen_hard)
    meta={'status':'passed','learned_rows':621,'learned_hard':125,'learned_tasks':20,'learned_demos':40,
       'unseen_rows':619,'unseen_hard':134,'unseen_tasks':10,'unseen_demos':100,
       'learned_raw_memory_error_q80':seen_threshold,'unseen_hard_source_threshold':hard_threshold,
       'norm_stats':str(norm),'norm_sha256':norm_sha,'checkpoint':str(CHECKPOINT),
       'corrected_image_pixel_rotation_check':'all 619 rows, both cameras, exact',
       'LIB1_asis_equals_legacy_k1':True,'fast_rot180_equals_corrected_k1':True,
       'scale_only_02_pi05_equals_corrected_k1':True,'controls':controls,
       'primary_unseen':primary,'seed':SEED,'unseen_noise_offset':len(rows['task']),
       'noise':'fold_in(key(4101), LIBERO40 row id); unseen uses 26581 + unseen row id',
       'steps':10,'batch':4,'normalization':'official Q01/Q99 affine; no extra clipping or gripper threshold'}
    save_json(out/'preflight.json',meta);log(out,'preflight_complete',unseen=primary)


def infer(out,batch=4):
    import torch
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    selection=load(out/'selection.npz');rows=data_rows(RUN/'rows');unseen=data_rows(UNSEEN)
    spec=importlib.util.spec_from_file_location('pi05_reference_base',HERE.parent/'pi05_memory/run.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    log(out,'policy_load_start')
    model,config=base.load_policy(CHECKPOINT)
    fn=module_jit(model.sample_actions,static_argnames=('num_steps',))
    def predict(source,ids,offset=0):
        obs=base.obs_batch(source,ids)
        keys=jax.vmap(lambda r:jax.random.fold_in(jax.random.key(SEED),r))(jnp.asarray(offset+ids,jnp.uint32))
        noise=jax.vmap(lambda k:jax.random.normal(k,(model.action_horizon,model.action_dim)))(keys)
        return np.asarray(fn(jax.random.PRNGKey(0),obs,num_steps=10,noise=noise))[:,:10,:7].astype(np.float32)
    parity_path=out/'fresh_unseen_parity.json'
    if not parity_path.exists():
        ids=selection['unseen_row'][:8]
        fresh=np.concatenate([predict(unseen,ids[i:i+batch],len(rows['task'])) for i in range(0,len(ids),batch)])
        cached=load(out/'predictions/unseen_cached.npz')['actions'][:len(ids)]
        diff=np.abs(fresh-cached)
        save_npz(out/'predictions/fresh_unseen_check.npz',actions=fresh,row=ids,cached=cached)
        meta={'rows':len(ids),'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),
              'checkpoint':str(CHECKPOINT),'num_steps':10,'batch':batch,'noise_offset':len(rows['task']),
              'model_source':base.__file__,'sample_actions_module':type(model).__module__}
        meta['pass']=meta['max_abs']<.002
        save_json(parity_path,meta)
        if not meta['pass']:raise AssertionError(f'fresh corrected-cache parity failed: {meta}')
        log(out,'fresh_unseen_parity_passed',**meta)
    else:
        assert json.loads(parity_path.read_text())['pass']
    chunks=out/'predictions/learned_parts';chunks.mkdir(exist_ok=True)
    ids=selection['learned_row']
    for start in range(0,len(ids),batch):
        path=chunks/f'batch-{start:05d}.npz';use=ids[start:start+batch]
        if path.exists():
            np.testing.assert_array_equal(load(path)['row'],use);continue
        padded=np.pad(use,(0,batch-len(use)),mode='edge')
        pred=predict(rows,padded)[:len(use)]
        save_npz(path,actions=pred,row=use)
        log(out,'learned_inference',done=start+len(use),total=len(ids))
    parts=[load(chunks/f'batch-{start:05d}.npz') for start in range(0,len(ids),batch)]
    all_ids=np.concatenate([p['row'] for p in parts]);np.testing.assert_array_equal(ids,all_ids)
    if not (out/'predictions/learned.npz').exists():
        save_npz(out/'predictions/learned.npz',actions=np.concatenate([p['actions'] for p in parts]),row=ids,
            target=np.asarray(rows['actions'][ids,:,:7],np.float32),hard=selection['learned_hard'])
    save_json(out/'inference.json',{'status':'complete','learned_rows':len(ids),'batch':batch,
         'model_source':base.__file__,'public_checkpoint':str(CHECKPOINT),'num_steps':10,
         'noise_seed':SEED,'noise_size':[model.action_horizon,model.action_dim],
         'note':'last batch padded to four by repeating its final row; only real rows saved'})
    log(out,'inference_complete')


def report(out):
    if (out/'results.json').exists():raise FileExistsError(out/'results.json')
    pre=json.loads((out/'preflight.json').read_text())
    result={'status':'complete','metric':'normalized active7 mean squared error over 10 steps x 7 axes',
      'normalization':pre['normalization'],'norm_stats_sha256':pre['norm_sha256'],
      'evaluation_sets':{},'controls':pre['controls'],'fresh_unseen_parity':json.loads((out/'fresh_unseen_parity.json').read_text()),
      'protocol':{'checkpoint':str(CHECKPOINT),'steps':10,'seed':SEED,'noise':pre['noise'],
          'observations':'CMP3 identical row start images/state/instruction; corrected images',
          'learned_selection':'CMP3 saved row IDs; validation only, no memory episode overlap',
          'postprocessing':'none: no clipping, no grip sign threshold, model normalized space'}}
    for split,file in [('learned','learned.npz'),('unseen','unseen_cached.npz')]:
        z=load(out/'predictions'/file);metrics=summarize(z['actions'],z['target'],z['hard'])
        result['evaluation_sets'][split]={'tasks':pre[split+'_tasks'],'demos':pre[split+'_demos'],
          'rows':len(z['row']),'hard_rows':int(z['hard'].sum()),'metrics':metrics,
          'prediction_source':str(out/'predictions'/file),'source_sha256':sha(out/'predictions'/file)}
        rr=metric_rows(z['actions'],z['target'])
        save_npz(out/f'{split}_row_metrics.npz',row=z['row'],hard=z['hard'],**rr)
    result['diagnosis']={'six_axis_mismatch':'image orientation: LIB1 used exact legacy K1 prediction; same-noise corrected K1 gives section5 value',
      'noise_offset_difference':'main active7 table uses official global keys; section5 uses K1 keys, slightly different corrected predictions',
      'old_outputs_edited':False,'model_training':False}
    save_json(out/'results.json',result)
    lines=['実行: 完了','','# PI0REF：CMP3と同じ行・同じ物差しの通常π0.5','',
      '公開pi05_liberoの通常出力の先頭10ステップを、未圧縮の正解10ステップと比較。正規化は公式Q01/Q99。active7は全10×7成分のMSE、腕寄与は腕6軸二乗和/70、グリッパー寄与は残る1軸二乗和/70で、両者の和がactive7に一致する。予測をクリップしたり、グリッパーを符号化したりしない。','',
      '| 評価集合 | 対象 | 窓数 | active7 MSE | 腕寄与（6軸和/70） | グリッパー寄与 |','|---|---|---:|---:|---:|---:|']
    for split,label in [('learned','学習済み LIBERO-40'),('unseen','未学習 LIBERO-90')]:
        d=result['evaluation_sets'][split]
        for subset,m in d['metrics'].items():
            lines.append(f"| {label} | {subset} | {m['rows']} | {m['active7']:.9f} | {m['arm_contribution']:.9f} | {m['gripper_contribution']:.9f} |")
    lines+=['','学習済みは20タスク・40デモ・621窓（難125）、未学習は10タスク・100デモ・619窓（難134）。CMP3の保存行IDに一致し、学習済み側の全評価デモはvalidationかつメモリ外。難行もCMP3と一致する。集合ごとにタスク・難行の分布が異なるため、この差を同一タスク上の因果効果とは扱わない。','',
      '## 未学習の既存値と照合','',
      '主表は orientation-fix-01/pi05/eval_global.npz の修正後予測を利用。既存の .1413 / .2870 を表示精度で再現した。キャッシュと今回のtargetの行順・正解を照合し、さらに公開重みで8行を再生成して確認した。','',
      '| 同じ619窓のキャッシュ | 画像 | ノイズキー | active7 全体/難 | 腕寄与 全体/難 | グリップ寄与 全体/難 |','|---|---|---|---:|---:|---:|']
    for name,image,noise in [('corrected_global','修正後','26581+row'),('corrected_k1','修正後','row'),
      ('legacy_k1','修正前','row'),('LIB1_fast_asis','修正前','row'),('fast_rot180','修正後','row'),('legacy_global','修正前','26581+row')]:
        m=pre['controls'][name]['metrics'];a,b=m['all'],m['hard']
        lines.append(f"| {name} | {image} | {noise} | {a['active7']:.9f}/{b['active7']:.9f} | {a['arm_contribution']:.9f}/{b['arm_contribution']:.9f} | {a['gripper_contribution']:.9f}/{b['gripper_contribution']:.9f} |")
    lines+=['','## .114 と .192 の食い違い','',
      '原因は6軸の割り算ではなく、古い画像向きで生成した予測をLIB1が参照したこと。LIB1 eval_summary.json が指定する fast-query-01/production/predictions/pi05_eval_asis.npz は、orientation-fix-01/pi05/legacy_eval_k1.npz と全要素完全一致する。一方、fast-queryの pi05_eval_rot180.npz は、修正後のeval_k1.npzと全要素完全一致する。両者は同じrowキー・同じ状態/指示/正解で、画像だけが180度異なる。修正済み画像と旧画像の180度反転も全619行・両カメラの全画素で一致を確認した。','',
      'したがって、今回の「修正後画像・通常π0.5・10ステップ」手順に対応するのは09 §5の約 .114/.251（K1ノイズ）であり、LIB1の約 .192/.307 は修正前画像での結果。主表の .1413/.2870 は以前のofficial_normal_evalと同じglobalノイズを使うため、腕寄与はK1ノイズと僅かに異なる。画像の違いとノイズの違いを混同しない。6軸の単純平均を出したい場合は腕寄与に7/6を掛けるが、予備実験表にはその値を混ぜない。','',
      '## 再現性・資源・保存','',
      f"- 公開重み：`{CHECKPOINT}`。norm_stats SHA256 `{pre['norm_sha256']}`。LIBERO-40と修正済みLIBERO-90の行manifestが同じnorm_statsを参照することを確認。",
      '- RNGは固定：LIBERO-40は fold_in(key(4101), 元の行ID)、LIBERO-90は fold_in(key(4101), 26581+元の行ID)。各窓で画像・状態・指示の先頭観測だけを渡し、10回積分。教師行動は生成入力に含めない。',
      '- 学習済み側は通常π0.5を新規生成し、各4行を predictions/learned_parts/ に原子的保存して再開可能にした。未学習側はキャッシュを再利用し、8行だけ新規生成して照合。公開重みの更新なし。',
      f"- 新規8行照合の最大絶対差 {result['fresh_unseen_parity']['max_abs']:.9g}、平均 {result['fresh_unseen_parity']['mean_abs']:.9g}。詳細は fresh_unseen_parity.json。",
      '- 使用コマンド・固定コード・CPU/GPU制約はcontrol/に記録。既存の予測・文書は上書きしていない。',
      '- 指定された /work/docs/memory_pilot_gates.md は存在せず、依頼文の制約に従った。git commitなし。','',
      '結論：予備実験表に使う通常π0.5の参考値は上のactive7表であり、.114対.192の不一致は修正後／修正前画像の予測キャッシュの混在で説明できる。','']
    (out/'report.md').write_text('\n'.join(lines));(out/'DONE').write_text('complete\n')
    log(out,'report_complete',metrics=result['evaluation_sets'])


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','infer','report'])
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--batch',type=int,default=4)
    a=p.parse_args()
    if a.stage=='preflight':preflight(a.out)
    elif a.stage=='infer':infer(a.out,a.batch)
    else:report(a.out)


if __name__=='__main__':main()
