# 保存済み実験との再現確認

成果物は読み取り専用で参照します。`SKILL_MEMORY_ARTIFACTS` は実験成果物ルートを指定します。テストは成果物ルートへファイルを書きません。

## 実行コマンド

```bash
cd "$SKILLNET_REPO_ROOT"
source env/activate.sh
export SKILL_MEMORY_ARTIFACTS="$SKILLNET_STATE_DIR"/outputs/memory-pilot/pi05-memory40-20261104
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
taskset -c 8-23 nice -n 10 python -m pytest -q -s skill_memory_core/tests/test_parity.py
```

テスト内容は、(1) M2のrow ID列とMaxSim score、(2) bridge式および同一乱数の学習1 update損失、(3) 16 queryでadapterと元実装を照合し、保存予測との差も計測、(4) raw-M6 row_units、(5) fair ExpertのF1/F2 bootstrapとaction oracle値。

## 結果

| 確認 | 期待値 | 実測 | 判定 |
|---|---|---|---|
| M2 shortlist＋MaxSim | productionの548×64行ID完全一致、MaxSim score max差 `<1e-5` | 548×64 ID完全一致、最大score差 `0` | 一致 |
| bridge式 / 1 update損失 | bridge array完全一致、loss完全一致 | NumPy/JAX配列と更新lossが完全一致 | 一致 |
| bridge生成 | 16 queryでadapterが元実装と一致 | adapter−元実装の最大絶対差 `0`。一方、保存 `offline_predictions.npz` とは最大差 `0.00185347` | 実装 parity一致／保存値不一致 |
| Raw M6 | `raw-m6-noLong-k0p1` とrow_units完全一致、unit数32 | `row_units` 完全一致、unit数 `32` | 一致 |
| 指標 | fair F1/F2 mean・CIを保存済み値と一致、oracle `.02899/.03512` | F1 all/hard `−.0015712 [−.0025259,−.0006880]` / `−.0104951 [−.0167006,−.0042255]`。訂正版F2 all/hard `+.0000261 [−.0006940,+.0007798]` / `+.0001965 [−.0058050,+.0071002]`。oracle all/hard `.0289912/.0351222` | 一致 |
| scratchコピー単体性 | packageを空dirへコピーしCPU testとminimalを通す | リポジトリをsys.pathから除いた一時環境で `12 passed, 5 skipped`。コピーからimport・minimal成功 | 一致 |

## 解釈上の限界

保存予測の差は現行adapterと同じ入力・重みの元参照実装でも再現するため、coreのadapter差ではなく保存時の実行条件または成果物間の版差と考えられますが、原因は特定できていません。元の保存予測を一致したものとして扱わず、再現性上の未解決差として残します。F2は初回のrank 1–5混合shuffleではなく、訂正済みのTop-1候補集合置換を使います。CPU suiteの5 parity testは成果物環境変数なしでskipし、成果物指定時は別コマンドで5件を実行します。閉ループ評価への同等性はこの表からは言えません。
