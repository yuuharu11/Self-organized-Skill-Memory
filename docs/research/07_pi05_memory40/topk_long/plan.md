# LIBERO-10 Long の生行動 Top-K 生成後診断

## 問いと範囲

今回の既知タスク・オフライン行動誤差について、生行動メモリの候補が固定済み bridge expert の下書きとして利用できるか、また候補集合から正解を見ずに選べるかを診断する。閉ループ成功率の主張はしない。学習・重み更新・検索キー再学習は行わない。

## 固定条件

- 元データは `pi05-memory40-20261104/run-01`。Eval 行は `memory-gap-01` の1,318行。LIBERO-10 Long は task ID 0〜9で、`run-01/data_manifest.json` の各 instruction が `data/LIBERO/libero_10/*_demo.hdf5` の10デモ名と対応することを実行時に検証する。
- hard 行は通常π0.5の行ごとのactive7 MSEがDev episode-mean q80 **0.01549410422449** 以上。閾値を再推定しない。
- メモリは元の6,583行のまま。主条件は同task除外、副条件は同episode除外。候補は保存済み retrieval-4b-01 の keyファイルから取得し、全16候補を同じメモリ行集合上で選ぶ。
- 下書きは候補の `rows/actions` [10,32]。圧縮codecは使わない。Expertは既存 `bridge/001500`、10積分、t0=.2/.4、各行で共通noise seed 4101。
- キーは `wrist_state_contrastive`（主）、`current`、`codec_prior_b`、`pi05_pred_to_memory_action`（後二者は方策依存の参考）。各キーでtask/episode除外を比較する。
- K={1,4,8,16}。Top-16を一度生成し、Kはそのprefixから集計する。

## 選択器と対照

各Kについて生成前 `pre_best@K`、生成後oracle `post_best@K`、および正解を見ない `rank1`、生成出力のmedoid `consensus`、出力と対応下書きの距離が最小の `min_edit`、通常π0.5出力に最も近い `near_normal` を記録する。near_normalはπ0.5誤差への追従の懸念があるため参考値とする。

対照 `zero_K` はゼロ下書きで16個のnoise seed `4101+1009*i` を使い、`normal_K` は通常π0.5で同じseed列を使う。zero_Kではrank1・consensus・min_edit・near_normalを計算し、normal_Kではrank1・consensusを計算する。各Kで先頭K件を使う。比較対象の単発通常π0.5はseed 4101。

Shuffled診断はmemory-gapと同じ固定shuffle seed 20261104・別task source-row規則でsource rowを定める。そのsource rowの検索候補から、現在の問い合わせtaskを除外した最初の候補を選び、生行動下書きとして推論する。

## 指標と統計

各Long行、各t0、キー、除外条件について生成前/後の誤差と選択器誤差を保存する。主キーのK=16では、候補間の生成出力距離平均を候補下書き距離平均で割った出力感度を計算する。

全Long行とhard行ごとに、episode単位・5,000反復bootstrapの95%区間を出す。

- memory `post_best@K` − `zero_K post_best@K`
- 各memory selector − 対応するzero_K selector
- 各memory selector − 単発通常π0.5

また主キーK=1 rank1のMSEをretrieval-4b-01の保存済み値と、zero_Kの先頭noise seedをmemory-gapのBと照合する。

## 事前判定

- **G1 生成側**：主キー・同task除外・hard行で、K≤8のいずれかにおいて `post_best@K(memory) − post_best@K(zero_K)` の95%区間上端が0未満なら候補を生成器が活かせる。満たさなければ「生成側を直すのが先」とする。
- **G2 選択側**：G1を満たし、正解を見ないselectorのいずれかでmemory−zero_Kの95%区間上端が0未満なら、実行時選択の見込みありとする。
- 出力感度がほぼ0、またはshuffledの性能を主キー条件と区別できなければ「下書きを無視している」と記す。
- 区間が0をまたぐ結果は「同等」と呼ばない。

## 実行

スモークは `topk-long-01/smoke-final/` に少数のhard Long行を保存する。本番は固定snapshotから以下を実行し、出力rootを `topk-long-01/` とする（root直下にあるsmoke*ディレクトリは保持し、rootファイルは新規作成する）。

```bash
cd "$SKILLNET_REPO_ROOT" && source env/activate.sh
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.20 \
  taskset -c 8-47 nice -n 10 python skill_moe/skillnet/examples/memory_pilot/pi05_memory/topk_long/run.py \
  --output .skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/topk-long-01
```
