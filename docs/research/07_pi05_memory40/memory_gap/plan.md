# π0.5 memory-gap：事前設定

段階2（オフライン A〜E 比較）。判定基準は承認済み `/tmp/claude-0/-work/574bf6f0-6862-4644-ab6a-213d9502a82e/scratchpad/msgs/plan_memory_gap.md` の段階2から固定し、結果を見て変更しない。

## 入力・評価行

- 読み取り元は `pi05-memory40-20261104/run-01` と `score-audit-01`。重みは `best_bridge.json` が指す `bridge/001500` のみを使用する。
- Eval は `calibrate.py::calibration_episodes` と同じ評価エピソードの1,318行、40タスク、80エピソード。全条件で同じ行を使う。
- 評価ノイズは draft-sensitivity と同じ seed 4101/4102、各行 epsilon は `fold_in(key(seed), row_id)`。bridge RNG は同じバッチ開始位置を使い、バッチ4・10積分。
- Dev の通常π0.5誤差は `calibration_errors.npz` の Dev 240行を再利用する。これは保存済み校正実行で同じ固定ノイズ規則により計算された値。Dev の各エピソードについて保存行の通常誤差平均を計算し、その80パーセンタイルを「難しい」閾値として固定する。Evalでは同じ2ノイズの通常π0.5行誤差を平均し、この閾値以上を難しい行とする。さらに、その難しい行のうち Eval 通常誤差上位5%を `hard_top5` とする。

## 条件と候補除外

| 条件 | Draft |
|---|---|
| A | 公式通常 `sample_actions`（下書きなし） |
| B | 正規化行動空間ゼロ |
| C | 保存済み現行 VLM `drafts.npy` |
| D-episode / D-task | oracle_action_mse 最小の許可メモリ行から保存 base を取り、Eval query の predicted residual で decode |
| D-raw-episode / D-raw-task | D と同じ候補行の `rows['actions']` 全32次元をそのまま draft として渡す |
| E | 正解行動を codec.encode した b と query predicted residual で decode |

`episode` 除外は同じ (task, episode) の候補だけを除外し、`task` 除外は同じ task の全候補を除外する。`oracle_meta.npz` の query/memory 行順を `query_meta.npz` と照合し、D の選択が許可候補内の oracle_action_mse 最小行であることを assert する。

B/C/D/D-raw/E を t0∈{.6,.4,.2} で評価し、A は通常生成のみ。D と D-raw は各除外方式ごとに候補選択を独立に行う。D/E は正解行動を使う診断用上限であり、実行可能な手法の性能とは扱わない。

## 集計・判定（固定）

- active7 正規化空間の最終行動 MSE（平均・中央値）、B との差のエピソード単位 percentile bootstrap 95% CI（5,000反復）。全行・hard・hard_top5を集計。
- 埋まった割合 `G=(B−X)/(B−E)` は難しい行について、各 t0 と除外条件で計算する。判定は難しい行の最良 t0 に対して行う。
- 難しい行で `G(D) >= 0.5` なら「有効な候補はメモリにあり、障害は検索」とし段階4-B。`G(D) < 0.2` なら「保存内容・表現が障害」とし段階4-A。0.2〜0.5は判定保留として段階4の大規模着手をしない。
- D-task が D-episode より大きく悪化すれば、候補が同じタスク内に偏ると記録。D-raw が D より明確に良ければ圧縮損失、D-raw も悪ければ座標系か区切りの問題を示唆する。C−B の95% CIが0をまたぐかも再確認として記録する。
- D/E はoracle診断であり、実行可能な手法の性能とは扱わない。閉ループ成功率はこの段階では用いない。

## 実装前の一致条件

- A（通常）と C（VLM draft）, t0=.6 の Eval MSE が `draft-sensitivity-01` の対応結果と一致する。
- D の候補行は oracle_action_mse の許可列の最小値と一致する。
