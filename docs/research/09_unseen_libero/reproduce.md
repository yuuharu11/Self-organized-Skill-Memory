# 再現手順：未学習 LIBERO タスクでの運動単位メモリ

この流れ（09）の主な結果を、新しく clone したリポジトリから作り直す手順です。
各段の出力が次の段の入力になります。表の順に実行してください。

> **画像の向き：** 未学習行の画像は、最初の作成（`oracle-01/unseen_rows/`）で180度回っていました。いまの `prepare_oracle.py` は閉ループと同じ向きに直して `unseen-rows-02/` に書き、全スクリプトがそこを読みます（[画像の向きの修正](image_orientation_fix.md)）。保存済みの段階 2〜9 の結果は旧い行で作ったものなので、この手順で作り直すと、画像を使う数値は文書と一致しません。

> **検証の範囲：** 新しい clone で、全テスト（`tests/`、`skill_memory_core/tests/`）が通ることと、各スクリプトが読み込めることを確認しました。
> 実験を最初から通し直した記録はありません。順序とコマンドは、保存済みの出力の依存関係と実行記録から整理しました。
> 学習と評価には GPU が要り、全段を通すと数日かかります。

> **保存済みの重み：** 段階 2 の Expert（`oracle-01/experts/`）と段階 3 のデノイズ Expert（`compressed-denoise-01/*/experts/`）の重みは、容量のため削除しました。下書き、行、特徴量、試行の記録、`results.json` は残っています。段階 7（`expert_refine.py`）や `MEM_EXPERT` を使う閉ループを回すときは、段階 2・3 の学習からやり直してください。
> 別環境に何をコピーすれば途中の段から始められるかは、[再現ガイド 7 節](../../reproduction.md#7-成果物の置き場所と別環境への持ち出し) にあります。

> **評価行の刻み（10 Hz ずれ）：** 未学習 619 行（`unseen-rows-02/`、`key-motion-01/final/data/unseen__*`）と旧 Dev 217 行は、HDF5 を1つおき（10 Hz）に取っています。メモリと学習済み行は 20 Hz（no-op 除去後の LeRobot フレーム）なので、10 ステップ窓の長さが2倍ずれています。
> 以後のオフライン評価は、段階 21 で作る `unseen20`／`dev20` と Dev v2 20 Hz（`dev-v2-01/data20/`）を使ってください。段階 14〜20 の 10 Hz 行で測った未学習の値は「旧（10 Hz ずれあり）」として扱います。閉ループは実際の 20 Hz 制御なので影響を受けません。

## 全体の流れ

`$U` は `$SKILLNET_STATE_DIR/outputs/memory-pilot/unseen-libero` です。

| # | 段階 | スクリプト | 出力（`$U` 以下） | 結果の文書 |
|---|---|---|---|---|
| 0 | LIBERO-40 の特徴・圧縮器・基準 Expert | `pi05_memory/run.py` | `$SKILLNET_PI05_ROOT/run-01/` | [現行アルゴリズムの再現手順](../../reproduce_core_path.md) の段階 1〜2 |
| 1 | 通常 π0.5 の閉ループ（基準） | `run.py` | `baseline-01/` | [実験一覧 1](experiments.md#1-通常-π05-の閉ループ基準) |
| 2 | 未学習デモの窓と VLM 特徴、既存 Expert の比較 | `prepare_oracle.py`、`oracle_experiment.py` | `unseen-rows-02/`（行）、`oracle-01/` | [実験一覧 2・3](experiments.md#2-メモリに未学習の動きはあるか正解で選ぶ上限) |
| 3 | 圧縮した単位の下書きとデノイズ | `compressed_denoise.py`、`low_freq_denoise.py` | `compressed-denoise-01/production/`、`production-lowfreq/` | [実験一覧 3](experiments.md#3-デノイズは下書きを直すか) |
| 4 | θ 回帰と LoRA | `theta_regression.py`、`theta_regression_lora.py` | `theta-regression-01/production/`、`lora/production/` | [実験一覧 4](experiments.md#4-向きと大きさθを推論時に決められるか) |
| 5 | K1：型を選ぶ入力の総当たり | `key_query_probe.py` | `key-query-01/production/` | [実験一覧 5](experiments.md#5-型をどう選ぶかクエリ) |
| 6 | θ のデノイズ | `lora_theta_on_selected.py`、`theta_flow.py` | `theta-flow-01/` | [実験一覧 4](experiments.md#4-向きと大きさθを推論時に決められるか) |
| 7 | Expert での仕上げ | `expert_refine.py` | `expert-refine-01/` | 同上 |
| 8 | K2：層の総当たり | `key_query_k2.py`、`k2_unseen_layers.py` | `key-query-02/` | [実験一覧 5](experiments.md#5-型をどう選ぶかクエリ) |
| 9 | K3・K4：クエリの改良 | `k3_query.py`、`k4_query.py` | `key-query-03/`、`key-query-04/` | 同上 |
| 10 | 可変長単位での検索 | `unit_retrieval.py` | `unit-retrieval-01/` | [実験一覧 5](experiments.md#5-型をどう選ぶかクエリ) |
| 11 | 単位メモリとクエリ | `cluster_pools.py` | `cluster-pools-01/` | [実験一覧 6](experiments.md#6-単位メモリの閉ループ) |
| 12 | 閉ループ | `memory_closed_loop.py` | `closed-loop-memory-01/` | 同上 |
| 13 | 向きを直した画像での取り直しと S1 の再採点 | `orientation_fix.py`、`scale_only_query.py refresh` | `orientation-fix-01/`、`scale-only-02/` | [画像の向きの修正](image_orientation_fix.md) |
| 14 | 運動キー MK1b | `key_motion.py`（`MK1B_ARCH_RUN=1`） | `key-motion-01/arch/` | [実験一覧 MK1b](experiments.md#mk1bキーの構造と目的) |
| 15 | クエリ MK1c・CT1 | `key_motion.py`（`MK1C_QUERY_RUN=1`）、`key_contrastive.py` | `key-motion-01/query/`、`key-contrastive-01/production/` | [MK1c](experiments.md#mk1cクエリ付き検索)、[CT1](experiments.md#ct1固定運動キーでの対照学習impl-3-の報告から転記) |
| 16 | KQ-final | `key_final.py`、`key_final_report.py` | `key-motion-01/final/` | [KQ-final](experiments.md#kq-finalmk1c-と-ct1-の食い違いの解消と最終設定) |
| 17 | CL1 閉ループ | `run_cl1_closed_loop.py`、`serve_expert_ex2.py`、`report_cl1.py` | `closed-loop-kq-01/` | [CL1](experiments.md#cl1運動キーexpert-の閉ループ) |
| 18 | EKQ | `expert_kq_runtime.py`、`run_ekq_closed_loop.py` | `expert-kq-01/` | [EKQ](experiments.md#ekqπ05-を使わない信頼度とグリッパー) |
| 19 | EX3 | `expert_ex3.py`、`serve_expert_ex3.py` | `expert-ex3-01/` | [EX3](experiments.md#ex3検索の外れ方に合わせて-expert-を学習し直すグリッパー込みπ05-なし) |
| 20 | DP1・TR1（分析） | `delta_probe.py`、`transfer_analysis.py` | `delta-probe-01/`、`transfer-analysis-01/` | [DP1](experiments.md#dp1記録時との差分から補正量が読めるか)、[TR1](experiments.md#tr1atomic-タスクで単位は別タスクから転用されているか既存ログの分析) |
| 21 | 20 Hz の評価行と Dev v2 | `vq_geometry.py rate20`、`dev_v2.py` | `vq-c-01/unseen20/`、`vq-c-01/dev20/`、`dev-v2-01/`（20 Hz は `data20/`） | [VQ-C](experiments.md#vq-c運動の選択に要る幾何の上限) |
| 22 | VQ-B | `vq_sources.py` | `vq-b-01/` | [VQ-B](experiments.md#vq-b画像と言語を分けて-film-で入れる) |
| 23 | VQ-C | `vq_geometry.py` | `vq-c-01/` | [VQ-C](experiments.md#vq-c運動の選択に要る幾何の上限) |
| 24 | VQ-A | `vq_film.py`、`vq_film_report.py` | `vq-a-01/` | 未記載（実行中） |

スクリプトは、特に書かない限り `skill_moe/skillnet/examples/memory_pilot/unseen_libero/` にあります。

**依存の要点：**

- 2 は 0 を読みます。3 は 0 と 2 を読みます。4 と 5 は 3 の `production/drafts` を読みます。5 は 4 の `production/features` も読みます。
- 6〜12 はすべて 5（`key-query-01/production/`）を読みます。
- 6 の `lora_theta_on_selected.py` は 4 の LoRA 出力を読みます。7、8 の `k2_unseen_layers.py`、12 の `check` は 6 の出力を読みます。
- 11 は 9 の `key-query-03/prep.npz` を読みます。12 で単位メモリを使うときは 11 の出力を指定します。
- 1〜2 の前に段階 0 を済ませてください。`run-01/` だけで約 92 GB を使います。

デノイズ前の比較（[実験一覧 2](experiments.md#2-メモリに未学習の動きはあるか正解で選ぶ上限)）は、データの準備（段階 2 の `prepare`）の後に下の「デノイズ前の比較」のコマンドで作ります。

## 準備

### 環境と変数

[再現ガイド](../../reproduction.md#2-別環境でのセットアップ) のセットアップを済ませ、リポジトリのルートで次を実行します。

```bash
source env/activate.sh
export SKILLNET_PI05_ROOT="$SKILLNET_STATE_DIR/outputs/memory-pilot/pi05-memory"   # 段階 0 の出力先
export SKILLNET_LIBERO_HDF5="$SKILLNET_STATE_DIR/data/LIBERO"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
U="$SKILLNET_STATE_DIR/outputs/memory-pilot/unseen-libero"
UL=skill_moe/skillnet/examples/memory_pilot/unseen_libero
CKPT="$SKILLNET_STATE_DIR/checkpoints/openpi-assets/checkpoints/pi05_libero"
```

| 環境変数 | 内容 | 未指定のとき |
|---|---|---|
| `SKILLNET_STATE_DIR` | データ・重み・出力の保存先 | `<リポジトリ>/.skillnet-state` |
| `SKILLNET_PI05_ROOT` | 段階 0 の出力（`run-01/` を含む） | `$SKILLNET_STATE_DIR/outputs/memory-pilot/pi05-memory40-20261104`（元の実験の run 名） |
| `SKILLNET_PI05_CHECKPOINT` | 公開 `pi05_libero` の重み | `$SKILLNET_STATE_DIR/checkpoints/openpi-assets/checkpoints/pi05_libero` |
| `SKILLNET_LIBERO_HDF5` | LIBERO の HDF5 | `$SKILLNET_STATE_DIR/data/LIBERO` |

すべてのスクリプトは、出力先を `$SKILLNET_STATE_DIR` から決めます。リポジトリの外に置く場合は、`source env/activate.sh` の前に `SKILLNET_STATE_DIR` を設定してください。

### データと重み

- `pi05_libero` の重み：[現行アルゴリズムの再現手順](../../reproduce_core_path.md#1-データと重み) の `gcloud storage cp`。
- LIBERO-90 のデモ（未学習10タスクと Dev 2タスク）と LIBERO-10：LIBERO 公式の `benchmark_scripts/download_libero_datasets.py` で `$SKILLNET_LIBERO_HDF5/libero_90`、`libero_10` に置きます。
- タスクの対応（未学習10タスク、Dev 2タスク）は `tasks.py` に固定してあります。

### タスクと分割

| 名前 | 中身 | 使い方 |
|---|---|---|
| メモリ | LIBERO-40 の400デモ（各タスク10本） | 単位と型の候補 |
| 学習行 | LIBERO-40 の `~validation` 23,304 行 | クエリ、θ、Expert の学習 |
| 保留タスク | LIBERO-40 のタスク [2, 7, 12, 17, 22, 27, 32, 37] | 時点・温度・方式の選択。候補からも除く |
| Dev | LIBERO-90 の2タスク（20デモ、113窓） | 一部の段の選択 |
| Eval | LIBERO-90 の未学習10タスク（100デモ、619窓） | 採点だけ。正解行動は他に使わない |

## 段階 1：通常 π0.5 の閉ループ

```bash
python $UL/run.py --output "$U/baseline-01" --checkpoint "$CKPT" --servers 4
```

10タスク×初期状態0〜9。期待値は 32/100。段階 12 はこの結果と同じ初期状態・seed で対にします。

## 段階 2：未学習デモの窓と VLM 特徴、既存 Expert の比較

```bash
python $UL/prepare_oracle.py prepare --hdf5-root "$SKILLNET_LIBERO_HDF5/libero_90"
python $UL/prepare_oracle.py extract --batch 4
python $UL/oracle_experiment.py drafts
python $UL/oracle_experiment.py train-queue --max-parallel 2
python $UL/oracle_experiment.py evaluate-all
python $UL/oracle_experiment.py finalize
```

`prepare_oracle.py` は `unseen-rows-02/` に Dev と Eval の窓、VLM の特徴を書きます。画像は HDF5 のフレームを `[::-1, ::-1]` で180度回してから入力変換に通します（閉ループと同じ）。
置き場所は `unseen_paths.py` の `UNSEEN_ROWS` 一か所で決まり、環境変数 `SKILLNET_UNSEEN_ROWS` で変えられます。既存のフォルダには書きません。
旧い（画像が逆さの）行 `oracle-01/unseen_rows/` は監査用に残してあり、どのスクリプトも読みません。
`oracle_experiment.py` は、正解で選んだ下書きを Expert で戻す17条件（[実験一覧 3](experiments.md#3-デノイズは下書きを直すか)）です。
`evaluate-all` が書く通常 π0.5 のオフライン予測（`official_normal_eval.npz`）と各条件の結果は、段階 3〜4 が読みます。

## 段階 3：圧縮した単位の下書きとデノイズ

```bash
python $UL/compressed_denoise.py run                 # 下書きと候補（production/drafts）、全条件の学習と評価
python $UL/low_freq_denoise.py energy
for c in L-vm-none-k3 L-vm-none-k3-wide L-vm-none-k5 L-vm-08-k3 L-raw-k3; do
  python $UL/low_freq_denoise.py train --condition $c
  python $UL/low_freq_denoise.py evaluate --condition $c
done
python $UL/low_freq_denoise.py report
```

出力の既定は `compressed-denoise-01/production/` です（`--output` で変えられますが、後の段はこの場所を読みます）。1条件ずつ回す場合は `train --condition <条件>` と `evaluate --condition <条件>` を使います。

## 段階 4：θ 回帰と LoRA

```bash
python $UL/theta_regression.py run --device cuda     # 特徴抽出、学習、評価（production/）
python $UL/theta_regression.py shape-select --split dev
python $UL/theta_regression.py shape-select --split eval
python $UL/theta_regression_lora.py run --device cuda   # Dev で選んだ条件で L-vlm / L-expert
```

LoRA の出力の既定は `theta-regression-01/lora/production/` です（`THETA_LORA_OUT` で変えられます）。

## 段階 5：K1（型を選ぶ入力の総当たり）

```bash
for s in extract train select calibrate-p8 evaluate report; do
  python $UL/key_query_probe.py --stage $s --output "$U/key-query-01" --threads 8
done
```

`train` は `--epochs 12 --batch-size 64 --device cuda` が既定の設定です。`extract` は中間層の特徴で約 25 GB を書きます。結果は `key-query-01/production/report.md`。

## 段階 6〜7：θ のデノイズと Expert での仕上げ

```bash
python $UL/lora_theta_on_selected.py                       # LoRA θ を K1 で選んだ型に当てる
python $UL/theta_flow.py --stage all --epochs 30 --device cuda
python $UL/expert_refine.py
```

`theta_flow.py` は `prepare → train → evaluate → report` を順に行います。`lora_theta_on_selected.py` を先に実行しないと、表から LoRA θ の行が抜けます。

## 段階 8：K2（層の総当たり）

```bash
for s in extract train select evaluate report; do
  python $UL/key_query_k2.py --stage $s --output "$U/key-query-02" --threads 16
done
python $UL/k2_unseen_layers.py
```

`train` と `evaluate` は `--batch-size 64 --device cuda` で実行しました。`train` は `--epochs 8`。

## 段階 9：K3・K4

```bash
python $UL/k3_query.py        # prep → 学習 → 評価（key-query-03/）
python $UL/k4_query.py        # key-query-03/prep.npz を使う（key-query-04/）
```

## 段階 10：可変長単位での検索

```bash
python $UL/unit_retrieval.py
```

## 段階 11〜12：単位メモリと閉ループ

```bash
python $UL/cluster_pools.py   # pool_unit_{none,04,08,16}.npz と query_unit_*.pt
P="$U/cluster-pools-01"; O="$U/closed-loop-memory-01"

# 窓の断片メモリ（K1 の履歴クエリ）
python $UL/memory_closed_loop.py run --output "$O/libero90" --checkpoint "$CKPT" --suite libero_90 --inits 10 --conditions memory

# 単位メモリ（τ ごと）
for name in unit_none unit_04 unit_08 unit_16; do
  MEM_POOL="$P/pool_$name.npz" MEM_CKPT="$P/query_$name.pt" \
    python $UL/memory_closed_loop.py run --output "$O/libero90_$name" --checkpoint "$CKPT" \
      --suite libero_90 --inits 10 --conditions memory
done

# π0.5 の Expert で下書きをデノイズ（t0 = .2 / .4 / .6）
for t0 in 0.2 0.4 0.6; do
  MEM_POOL="$P/pool_unit_08.npz" MEM_CKPT="$P/query_unit_08.pt" MEM_T0=$t0 \
    python $UL/memory_closed_loop.py run --output "$O/libero90_unit_08_dn${t0/./}" --checkpoint "$CKPT" \
      --suite libero_90 --inits 10 --conditions memory
done

python $UL/memory_closed_loop.py report --output "$O/libero90_unit_08" --suite libero_90
```

LIBERO-10 は `--suite libero_10 --inits 5` にします（各タスクと同じタスク由来の単位は自動で除かれます）。
`--conditions lowpass` は π0.5 の出力を DCT 3個に落とすだけの対照です。
環境変数 `MEM_EXPERT` に [実験一覧 3](experiments.md#3-デノイズは下書きを直すか) の Expert を指定すると、公開 Expert の代わりに使います。

**集計の注意：** LIBERO-10 の出力に `report --suite libero_10` を使うと、0/0 と表示されます。基準との対応づけが LIBERO-90 用のためです。LIBERO-10 の成功数は `memory/attempts/*/attempt-*/result.json` の `success` を数えてください。

## 段階 13：向きを直した画像での取り直しと S1 の再採点

LIBERO-40 側は正しい向きだったので、学習済みのクエリは学習し直しません。未学習行の画像から作る入力だけを取り直します。

```bash
python $UL/orientation_fix.py all        # π0.5 の予測（同じ固定ノイズ）と VLM 平均を Dev/Eval で（orientation-fix-01/）
O="$U/scale-only-02/production"
python $UL/scale_only_query.py refresh --out "$O" --src "$U/scale-only-01/production" --fix "$U/orientation-fix-01" --modes P0,P5,P6
python $UL/scale_only_query.py train --out "$O" --modes P0      # P0（VLM 平均だけ）を P6 と同じ学習・選択で
python $UL/scale_only_query.py evaluate --out "$O"
python $UL/scale_only_query.py correlations --out "$O"
python $UL/scale_only_query.py report --out "$O"
```

`orientation_fix.py` は、旧い行で K1 の π0.5 予測（差 5e-4 以内、float16 の丸め）と θ 回帰の VLM 平均（差 0）を再現できることを確かめてから、直した行で計算します。
`refresh` は scale-only-01 の LIBERO-40 側（プール、学習入力、教師、学習済みモデル、選択）をそのまま写し、Dev/Eval の VLM 平均と π0.5 の入力だけを差し替えます。

## 段階 14 以降の共通事項

段階 14〜24 のコマンドは、出力に残った `progress.log`・`plan*.json`・`*.sh` から整理したものです。通し直した記録はありません。
これらの段は、この表にない既存の出力も読みます：`key-shared-01/`（KEY2）、`key-grid-01/production-04/`（KEYGRID）、`seg-context-01/`（stride-5 キャッシュ）、`expert-ex1-01/`・`expert-ex2-01/production-05/`（EX1・EX2 の Expert）。これらの作り方はこの文書にまだ書いていません。
保存済みの `results.json` や `rows/*.json` には、重みの絶対パス（`/work/.skillnet-state/...`）が入っています。別の場所に出力をコピーして途中の段から始めるときは、その場所へのシンボリックリンクを置くか、作り直してください。

重い成果物はすべて `$SKILLNET_STATE_DIR` の下で、Git 管理外です。容量の目安：`key-motion-01/` 14 GB（`final/data/` 13 GB）、`key-contrastive-01/` 20 GB、`expert-ex2-01/production-05/` 42 GB、`expert-ex3-01/` 90 GB（`train-G1`・`train-G2` 各 45 GB）、`vq-a-01/` 27 GB、`vq-b-01/` 8.6 GB、`dev-v2-01/` 2.8 GB、`vq-c-01/` 263 MB（`hdf5_states/` 235 MB）、`expert-kq-01/` 97 MB、`closed-loop-kq-01/` 31 MB。

## 段階 14：運動キー MK1b

```bash
MK1B_ARCH_RUN=1 python $UL/key_motion.py --output "$U/key-motion-01/arch"
```

X1/X1s/X5 と位置・目的の総当たり（seed 1、4 epoch）。後の段は `arch/x1_absolute_dist_tau0.01.pt` を固定キーとして読みます。`--output` は新しいフォルダでないと止まります。
MK1 本体（`key-motion-01/m1.pt`）は `python $UL/key_motion.py --output "$U/key-motion-01"` と `--stage2` で作りました（報告は `key-motion-01/report.md`）。

## 段階 15：クエリ MK1c・CT1

```bash
MK1C_QUERY_RUN=1 python $UL/key_motion.py --output "$U/key-motion-01/query"
python $UL/key_contrastive.py --output "$U/key-contrastive-01" --smoke            # data/ のキャッシュを作る
python $UL/key_contrastive.py --output "$U/key-contrastive-01/production" --data-cache "$U/key-contrastive-01/data"
```

CT1 は段階 14 の X1 キーを読みます。`--smoke` の結果（`key-contrastive-01/results.json`）は本番ではありません。

## 段階 16：KQ-final

```bash
O="$U/key-motion-01/final"
python $UL/key_final.py prep
for p in A B D H S; do python $UL/key_final.py rows --plan "$O/plan_$p.json"; done
python $UL/key_final.py keygrid --plan "$O/plan_KG.json"
python $UL/key_final_report.py
```

`plan_*.json` は出力に残してあります（[evidence/kq_final/](evidence/kq_final/) にはありません。保存済みの出力からコピーしてください）。本番は各 plan を `--names` で分けて複数の GPU プロセスで回しました。
`best/` は選ばれた行の `rows/<名前>.pt` をそのままコピーしたもの、`export/` は `key_final.export_row(path, ctx, dest, name)` を Python から呼んで書いたものです（CLI にはありません。**コマンドは未確認**）。

## 段階 17：CL1 閉ループ

```bash
E="$U/expert-ex2-01/production-05"; O="$U/closed-loop-kq-01"; mkdir -p "$O/expert-select"
for f in dev-selection.npz retrieval train-A train-B; do ln -s "$E/$f" "$O/expert-select/$f"; done
python -c "import sys,json,pathlib;sys.path.insert(0,'$UL');import expert_ex2 as e;print(json.dumps(e.repair_dev_checkpoint_selection(pathlib.Path('$O/expert-select'))['selected_by_condition'],indent=1))"
python $UL/run_cl1_closed_loop.py parity
python $UL/run_cl1_closed_loop.py smoke
python $UL/run_cl1_closed_loop.py run --conditions C1a,C1b,C3,C2 --workers 10 --parallel --fraction .22
python $UL/run_cl1_closed_loop.py summarize
python $UL/report_cl1.py
```

LIBERO-90 Eval 10タスク×10試行。段階 16 の `best/`・`export/` と EX2 production-05 を読みます。本番では `SKILLNET_SUBTASK_GENERATE=0`、`OMP_NUM_THREADS=1` にしました（`closed-loop-kq-01/launch.sh`）。

## 段階 18：EKQ

```bash
for q in C1a C1b; do for f in 0 1 2 3 4; do
  python $UL/expert_kq_runtime.py crossfit --query $q --folds $f &
done; done; wait
for c in retrieve fit fit-gripper dev-grid select bundle eval report; do python $UL/expert_kq_runtime.py $c; done
python $UL/run_ekq_closed_loop.py smoke
python $UL/run_ekq_closed_loop.py run --conditions E1a,E1b --workers 10
python $UL/run_ekq_closed_loop.py run --conditions E1a_pg --workers 10
python $UL/run_ekq_closed_loop.py summarize
```

段階 17 の `closed-loop-kq-01/expert-select/retrieval` を読みます。

## 段階 19：EX3

```bash
python $UL/expert_ex3.py measure
python $UL/expert_ex3.py prepare
XLA_PYTHON_CLIENT_MEM_FRACTION=.45 python $UL/expert_ex3.py train --name G1 --rank-scale 1.0 --grip-weight 1.0 --microbatch 8
XLA_PYTHON_CLIENT_MEM_FRACTION=.42 python $UL/expert_ex3.py train --name G2 --rank-scale 5.0 --grip-weight 1.0 --microbatch 8
for c in select devgrid offline; do python $UL/expert_ex3.py $c; done
python $UL/expert_ex3.py closed --conditions X3a,X3b --workers 10 --fraction .15
python $UL/expert_ex3.py report
```

段階 18 の `expert-kq-01/heads/`・`retrieval/` を `ekq-frozen/` に写して使います。`train` は1本 45 GB の checkpoint を書きます。

## 段階 20：DP1・TR1（分析）

```bash
python $UL/delta_probe.py --device cuda      # 本番のデバイス指定は未確認（既定は cpu）
python $UL/transfer_analysis.py
```

DP1 は KEY2（`key-shared-01/attempt-27-production/final/p5/`）を読みます。最初の版は不具合で差し替え、旧出力は `delta-probe-01/superseded-keybug/` にあります。
TR1 は学習しません。単位メモリ 43/100、EX1、EX2、RoboCasa（`robocasa-memory/latent-skill-01/`）の既存の試行ログを読みます。

## 段階 21：20 Hz の評価行と Dev v2

```bash
python $UL/vq_geometry.py rate20 --split unseen dev     # vq-c-01/unseen20/、dev20/
python $UL/dev_v2.py build                              # dev-v2-01/data/（10 Hz、VLM トークン込み）
python $UL/dev_v2.py finalize
python $UL/dev_v2.py build20                            # dev-v2-01/data20/（20 Hz。主に使うのはこちら）
python $UL/vq_film.py validate                          # dev-v2-01/validation20.json（Dev v2 と unseen20 の順位一致）
```

`rate20` は LIBERO-90 の HDF5 から、開始＝2×frame 以上で最初の非 no-op ステップ、履歴＝直前 10 の非 no-op ステップ、目標＝次の 10 ステップで行を作り直します。読み込みは `vq_geometry.load_rate_matched('unseen20' | 'dev20')`、`dev_v2.load_dev_v2_20()`。
10 Hz 版の `dev-v2-01/validation.json` を作ったコマンドは**未確認**です（いまの CLI にはありません）。

## 段階 22：VQ-B

```bash
python $UL/vq_sources.py extract --splits learned unseen dev eh dev2 unseen20 dev20 dev2_20
python $UL/vq_sources.py extract --splits eh learned --res128          # I1r（LIBERO-40 を 128px に縮めて抽出）
python $UL/vq_sources.py train --tag sweep20 --seeds 4101               # 全条件、Dev v2 20Hz で選択
python $UL/vq_sources.py train --tag final20 --seeds 4102 4103 --rows none L1b L1b+I1r I1r I2
python $UL/vq_sources.py ablate --tag final20 --rows L1b L1c I1r L1b+I1r I2 L1a
python $UL/vq_sources.py analyze
python $UL/vq_sources.py report
```

seed 4101 の `final20/` は `sweep20/` からのコピーです。`ablate` の行と `--tag` の指定は progress.log からの推定で、**未確認**です。`sweep_10hz_dev/`・`final_10hz_dev_aborted/` は 10 Hz の Dev で選んだ旧版です。

## 段階 23：VQ-C

LIBERO-40 の sim 状態は LeRobot 版にないので、HF の公式 HDF5（`yifengzhu-hf/LIBERO-datasets`）から `actions`・`states`・`obs/ee_pos` だけを HTTP の範囲読み出しで取ります（画像は読みません、約 235 MB）。リダイレクト先の CDN URL を一度解決し、`fsspec` の https ファイルを `h5py` で開きます（要ネットワーク）。

```bash
python $UL/vq_geometry.py fetch40          # vq-c-01/hdf5_states/task*.npz（40タスク）
python $UL/vq_geometry.py align40          # LeRobot と HDF5 を no-op 除去後の行動の完全一致で対応づけ（440/440）
python $UL/vq_geometry.py geometry --split unseen dev learned eh
python $UL/vq_geometry.py geo_devv2        # Dev v2 20Hz の幾何
O="$U/vq-c-01"
python $UL/vq_geometry.py rows --plan "$O/plan_v2.json" --dev-split devv2   # 主表（Dev v2 20Hz で選択）
python $UL/vq_geometry.py compare
python $UL/vq_geometry.py ablate --names G_film_full_v2_s1 G_film_focus_v2_s1
python $UL/vq_geometry.py final
```

`plan_step1.json`（旧 Dev）と `plan_d20.json`（dev20）は途中の版です。`rows` の `--dev-split` の値と `eval20` の呼び方は**未確認**です（plan の `dev` 欄でも指定されます）。

## 段階 24：VQ-A

```bash
O="$U/vq-a-01"
python $UL/vq_film.py prep                                   # C1b 学習行の π0.5 トークン（data/、約 26 GB）
for p in grid extra c1b final; do python $UL/vq_film.py rows --plan "$O/plan_$p.json"; done
python $UL/vq_film.py rescore --names <KQ-final の行名>      # rescore_kq.json（行名は未確認）
python $UL/vq_film_report.py
```

`rows` は `rows20/` に書き、`.lock` で複数プロセスに分けられます（本番は4並列）。`rows/` は 10 Hz の Dev で選んだ最初の格子で、使いません。**この段は実行中で、結果と報告はまだありません。**

## デノイズ前の比較（実験一覧 2）

正解で単位を選ぶ上限の比較です。GPU は要りません。出力は `$U/draft-reconstruction-01/` の各 JSON です。

```bash
python $UL/draft_reconstruction.py
for s in variable_length_error variable_memory_oracle variable_shape_memory dct_tolerance_merge \
         dct_skill_draft_eval stored_units_eval optimal_units_eval compression_compare \
         dct_coefficients_eval motion_core_eval; do
  python $UL/$s.py
done
python $UL/segment_demo.py     # 切り出しの動画（segmentation_demo/）
```


## KQ2 / KQ2-CL

以下は保存済みの重み・特徴・単位キャッシュがある環境での再実行例です。大きな重みや特徴キャッシュは Git evidence に含めていません。KQ2-CL は LIBERO-90/10 の環境依存があるため、初めに smoke と parity を実行してください。すべての出力は `$SKILLNET_STATE_DIR/outputs/memory-pilot/unseen-libero/` 以下に置きます。

```bash
source env/activate.sh
export SKILLNET_STATE_DIR="${SKILLNET_STATE_DIR:-$PWD/.skillnet-state}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
U="$SKILLNET_STATE_DIR/outputs/memory-pilot/unseen-libero"
P=skill_moe/skillnet/examples/memory_pilot/unseen_libero
PY="$SKILLNET_STATE_DIR/envs/skillnet/bin/python"

# KQ2: per-step streams/VLM cache, key+query comparisons, report
nice -n 10 taskset -c 8-15 $PY $P/kq2_data.py build
nice -n 10 taskset -c 8-15 $PY $P/kq2_data.py state_check
nice -n 10 taskset -c 8-15 $PY $P/kq2_run.py baselines
nice -n 10 taskset -c 8-15 $PY $P/kq2_run.py keys --plan "$U/kq2-01/plan_keys.json"
nice -n 10 taskset -c 8-15 $PY $P/kq2_run.py keyq --plan "$U/kq2-01/plan_keyq.json"
nice -n 10 taskset -c 8-15 $PY $P/kq2_run.py query --plan "$U/kq2-01/plan_pre.json"
nice -n 10 taskset -c 8-15 $PY $P/kq2_run.py query --plan "$U/kq2-01/plan_query.json"
nice -n 10 taskset -c 8-15 $PY $P/kq2_run.py ablate --plan "$U/kq2-01/plan_query.json"
nice -n 10 taskset -c 8-15 $PY $P/kq2_report.py

# KQ2-CL: retrieve -> observations/pi0.5 -> Expert -> heads -> closed-loop/report
for stage in folds retrieve measure; do nice -n 10 taskset -c 8-15 $PY $P/kq2cl_retrieve.py $stage; done
for stage in build pi05; do nice -n 10 taskset -c 8-15 $PY $P/kq2cl_obs.py $stage; done
nice -n 10 taskset -c 8-15 $PY $P/kq2cl_expert.py prepare --w-mem .3 --w-dev .7
XLA_PYTHON_CLIENT_MEM_FRACTION=.40 nice -n 10 taskset -c 8-15 $PY $P/kq2cl_expert.py train --name K
nice -n 10 taskset -c 8-15 $PY $P/kq2cl_heads.py train
XLA_PYTHON_CLIENT_MEM_FRACTION=.30 nice -n 10 taskset -c 8-15 $PY $P/kq2cl_expert.py select
XLA_PYTHON_CLIENT_MEM_FRACTION=.35 nice -n 10 taskset -c 8-15 $PY $P/kq2cl_expert.py curves --tags K
nice -n 10 taskset -c 8-15 $PY $P/kq2cl_heads.py policy
nice -n 10 taskset -c 8-15 $PY $P/kq2cl_heads.py offline
SKILLNET_SUBTASK_GENERATE=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.20 nice -n 10 taskset -c 8-15 $PY $P/kq2cl_closed.py run --suite libero90 --conditions A,B,Aconst,Bconst --fraction .20
SKILLNET_SUBTASK_GENERATE=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.20 nice -n 10 taskset -c 8-15 $PY $P/kq2cl_closed.py run --suite libero10 --conditions A,B,Aconst --fraction .20
nice -n 10 taskset -c 8-15 $PY $P/kq2cl_closed.py summarize
nice -n 10 taskset -c 8-15 $PY $P/kq2cl_report.py
```

KQ2-CL の原実験は π0.5 基準の実行計画を変更し、保存済み baseline を再利用した。Libero-90 は10試行/タスク、LIBERO-10 も今回10試行/タスク。π0.5 の途中評価93試行は `closed-loop-pi05-aborted/` に残し、主比較には使っていない。結果 JSON 内の元環境パスは evidence では書き換えていない。再現確認では `$SKILLNET_STATE_DIR` 内の入力キャッシュとチェックポイントの存在を確認すること。

## 確認のしかた

各段の結果は、出力の `report.md`（または `summary.json`）に書かれます。研究ノートの表と比べてください。
主な表の元データは [evidence/](evidence/) にあります。

| ファイル | 対応する表 | 作った元（`$U` 以下） |
|---|---|---|
| `closed_loop_summary.json`、`closed_loop_trials.csv` | README 主な結果 1（閉ループ）。試行ごとの成否なので対の比較もできる | `baseline-01/production/`、`closed-loop-memory-01/*/{normal,lowpass,memory}/rollouts.jsonl` |
| `offline_mse_queries.json` | 主な結果 2 の直前保持、π0.5、π0-FAST、random、VLM 平均、π0.5 最近傍、「正解で選び大きさも正解」 | `fast-query-01/production/results.json` |
| `offline_mse_scale_only.json` | 主な結果 2 の P5、大きさを π0.5 から、正解で選ぶ、型・向き・大きさを正解、R-so3 | `scale-only-02/production/results.json` |
| `draft_reconstruction/*.json` | 主な結果 3（圧縮、借りる単位、可変長、τ でまとめる、デノイズ） | `draft-reconstruction-01/` |
| `evidence/kq2/`、`evidence/kq2_cl/` | KQ2 検索比較・KQ2-CL 閉ループ | `kq2-01/`、`kq2-cl-01/` |

閉ループの成功数は `rollouts.jsonl` の `success` を数えたものです（LIBERO-10 の `summary.json` は 0/0 と出るため使いません。段階 11〜12 の注意を参照）。
学習の乱数とGPUの違いで小数第3位は揺れます。結論（どれが基準より良いか、どれが差の何割を埋めるか）が変わるかで判断してください。

## T0X T4 / T5：Expert 修正量と上位候補不一致

この段は KQ2-CL の検索・Expert・heads・closed-loop キャッシュに加えて T0X T1〜T3 の比較試行を参照します。チェックポイントを再学習せず、T4/T5 は各条件を順番に実行します。T4 は Dev v2 の t_probe=.30 から校正、T5 は Dev v2 のみで D_b を校正します。評価は LIBERO-90/10 各10タスク×各10試行、各タスク内で試行 i の seed は `7+i`、初期状態 index も i（T1 と同じ）、wait=10、replan=5、最大520制御 step。通常 π0.5 と比較試行は保存済み結果を使います。

```bash
cd /work && source env/activate.sh
export SKILLNET_STATE_DIR="${SKILLNET_STATE_DIR:-$PWD/.skillnet-state}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false SKILLNET_SUBTASK_GENERATE=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
U="$SKILLNET_STATE_DIR/outputs/memory-pilot/unseen-libero"
P=skill_moe/skillnet/examples/memory_pilot/unseen_libero
PY="$SKILLNET_STATE_DIR/envs/skillnet/bin/python"

# T4: Dev 校正、各条件2試行 smoke、続いて各スイートで本番。
XLA_PYTHON_CLIENT_MEM_FRACTION=.20 nice -n 10 taskset -c 24-31 \
  "$PY" "$P/t0x_t4.py" all --workers 6 --fraction .20 --cpus 24-31,32-39 --batch 1

# T5: top-10 disagreement 校正、条件ごとに smoke 後、順番に6条件を実行。
XLA_PYTHON_CLIENT_MEM_FRACTION=.20 nice -n 10 taskset -c 24-31 \
  "$PY" "$P/t0x_t5.py" all --workers 6 --fraction .20 --server-cpus 24-39
```

完了ファイルは `$U/t0x-01/t4/DONE` と `$U/t0x-01/t5/DONE`。出力の元 JSON には実験機の絶対パスが残る場合があります。重みと大きな特徴キャッシュは Git evidence に含みません。

## KC1 / KC2 / XF1 / XF3

KQ2-CL の部品（クエリ、Expert K1:500、閉ループのプロトコル）と T5 の校正を使います。KC1 の保存係数は `bank__coefficients.npy` をそのまま読みます。閉ループは KQ2-CL と同じく、各タスク10試行、seed 7+i、wait 10、replan 5、最大520 step です。

```bash
cd /work && source env/activate.sh
export SKILLNET_STATE_DIR="${SKILLNET_STATE_DIR:-$PWD/.skillnet-state}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false SKILLNET_SUBTASK_GENERATE=0
P=skill_moe/skillnet/examples/memory_pilot/unseen_libero
PY="$SKILLNET_STATE_DIR/envs/skillnet/bin/python"

# KC1: オフライン（36設定、Dev v2 で選択）→ 閉ループ（LIBERO-90 と LIBERO-10）
"$PY" "$P/kc1_run.py" offline
"$PY" "$P/kc1_closed.py" all --workers 6 --fraction .20

# KC2: 3 seed を学習 → Dev 選択と評価 → 閉ループ
for s in 1 2 3; do "$PY" "$P/kc2_run.py" train-seed --seed $s; done
"$PY" "$P/kc2_eval.py" evaluate
"$PY" "$P/kc2_closed.py" all --workers 6 --fraction .20

# XF1: 5分割の交差検索と信頼度の相関（CPU）
CUDA_VISIBLE_DEVICES='' "$PY" "$P/xf1_run.py"

# XF3: グリッパーデコーダ 12条件×3 seed（CPU）
CUDA_VISIBLE_DEVICES='' "$PY" "$P/xf3_run.py" all
```

出力は `$SKILLNET_STATE_DIR/outputs/memory-pilot/unseen-libero/{kc1-01,kc2-01,xf1-01,xf3-01}/` に書かれ、完了すると `DONE` が置かれます。Git の evidence には、report・集計 JSON・全試行の `trials.csv` だけを入れています。

## テスト

データや重みなしで、合成入力で確かめるテストです。

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu python -m pytest -q \
  tests/unseen_libero/
```

保存済みの出力がない環境では、出力に依存する一部のテストは skip になります。
