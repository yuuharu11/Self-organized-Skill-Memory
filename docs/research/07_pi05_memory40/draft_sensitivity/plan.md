# 修正 Expert の下書き依存性：事前設定

## 問いと判定

修正 Expert の改善が、渡した下書きの内容に依存するか、action expert の追加学習だけで通常方策を改善したかをオフライン行動誤差で切り分ける。閉ループ成功率はこの診断に使わない。oracle と shuffled/zero の差が小さければ下書き内容への依存は弱く、追加学習の効果が中心と解釈する。差が明確なら正しい運動を渡す経路に価値があると解釈する。

## 固定データ・重み

- 読み取り元: `.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01/`。
- `rows['validation']` のうち、`calibrate.py::calibration_episodes` と同じ seed `20261105 + task_id` で分けた Eval エピソード。1,318行を固定する。Dev 側は含めない。
- 公式 `pi05_libero` を通常推論の基準にする。提案条件では `best_bridge.json` が示す `bridge/001500` のみを action-expert leaf に上書きする。`latest_bridge.json` は選択に使わない。
- Active 7 次元、正規化行動空間、10 denoise steps。
- まず校正保存行から16行で形・推論一致・速度を確かめる。見積時間が40分を超えるときはタスク層別400行に固定縮小して全条件を回す。

## 条件

`t0=.6` を主条件として oracle、bprior_direct、vlm_retrieved、bprior_retrieved、shuffled、zero、normal を実行する。oracle は `codec.encode(A,s)` の b と保存済み query predicted residual を decode。bprior_direct は `codec.prior(pooled,s)` の b/r を decode。vlm_retrieved は保存済み `drafts.npy`。bprior_retrieved は score-audit の episode-excluded Top-1 行の保存 base と query の predicted residual。shuffled は別タスクから固定 seed で選んだ VLM 下書き。zero は正規化空間のゼロ。normal は expert overlay 前の公式 `sample_actions`。

下書き依存の開始時刻効果は oracle、bprior_direct、shuffled のみ `t0=.4,.2` も測る。各行で全条件・全 t0・normal に同じ固定 epsilon を渡す。epsilon は calibrate と同じ `fold_in(key(noise_seed), row_id)`、`noise_seed∈{4101,4102}`。各チャンクの bridge RNG は `fold_in(key(noise_seed), batch_start)`、baseline RNG は `key(0)`。

## 指標と集計

- 各条件の行動 MSE は 10×7 active 次元の平均。2 noise の予測を行単位で平均して集計。
- 平均、中央値、MSE≤.01 の割合。normal に対する差は Eval episode 単位の percentile bootstrap 95% CI（5,000反復、seed 20261104）。
- 下書き MSE と最終 MSE の Spearman、下書き誤差の四分位別最終誤差、最終行動と下書きの active7 RMS 距離。
- normal は下書き指標なし。

実行コードは `examples/memory_pilot/pi05_memory/draft_sensitivity/`、出力は `.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/draft-sensitivity-01/`。
