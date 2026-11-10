# 再現手順：未学習 LIBERO タスクでの運動単位メモリ

この流れ（09）の主な結果を、新しく clone したリポジトリから作り直す手順です。
各段の出力が次の段の入力になります。表の順に実行してください。

> **検証の範囲：** 新しい clone で、全テスト（`tests/`、`skill_memory_core/tests/`）が通ることと、各スクリプトが読み込めることを確認しました。
> 実験を最初から通し直した記録はありません。順序とコマンドは、保存済みの出力の依存関係と実行記録から整理しました。
> 学習と評価には GPU が要り、全段を通すと数日かかります。

## 全体の流れ

`$U` は `$SKILLNET_STATE_DIR/outputs/memory-pilot/unseen-libero` です。

| # | 段階 | スクリプト | 出力（`$U` 以下） | 結果の文書 |
|---|---|---|---|---|
| 0 | LIBERO-40 の特徴・圧縮器・基準 Expert | `pi05_memory/run.py` | `$SKILLNET_PI05_ROOT/run-01/` | [現行アルゴリズムの再現手順](../../reproduce_core_path.md) の段階 1〜2 |
| 1 | 通常 π0.5 の閉ループ（基準） | `run.py` | `baseline-01/` | [通常方策の結果](baseline_results.md) |
| 2 | 未学習デモの窓と VLM 特徴、既存 Expert の比較 | `prepare_oracle.py`、`oracle_experiment.py` | `oracle-01/` | [実行前計画](oracle_plan.md)、[表現上限のオフライン比較](oracle_results.md) |
| 3 | 圧縮した単位の下書きとデノイズ | `compressed_denoise.py`、`low_freq_denoise.py` | `compressed-denoise-01/production/`、`production-lowfreq/` | [圧縮表現から始めるデノイズ](compressed_denoise.md) |
| 4 | θ 回帰と LoRA | `theta_regression.py`、`theta_regression_lora.py` | `theta-regression-01/production/`、`lora/production/` | [θ回帰と型選択](theta_regression.md) |
| 5 | K1：型を選ぶ入力の総当たり | `key_query_probe.py` | `key-query-01/production/` | [型の選び方](motion_query.md) |
| 6 | θ のデノイズ | `lora_theta_on_selected.py`、`theta_flow.py` | `theta-flow-01/` | [θ のデノイズと Expert での仕上げ](theta_flow.md) |
| 7 | Expert での仕上げ | `expert_refine.py` | `expert-refine-01/` | 同上 |
| 8 | K2：層の総当たり | `key_query_k2.py`、`k2_unseen_layers.py` | `key-query-02/` | [型の選び方](motion_query.md) |
| 9 | K3・K4：クエリの改良 | `k3_query.py`、`k4_query.py` | `key-query-03/`、`key-query-04/` | 同上 |
| 10 | 可変長単位での検索 | `unit_retrieval.py` | `unit-retrieval-01/` | [可変長単位での検索](unit_retrieval.md) |
| 11 | 単位メモリとクエリ | `cluster_pools.py` | `cluster-pools-01/` | [閉ループ評価](closed_loop_unit_memory.md) |
| 12 | 閉ループ | `memory_closed_loop.py` | `closed-loop-memory-01/` | 同上 |

スクリプトは、特に書かない限り `skill_moe/skillnet/examples/memory_pilot/unseen_libero/` にあります。

**依存の要点：**

- 2 は 0 を読みます。3 は 0 と 2 を読みます。4 と 5 は 3 の `production/drafts` を読みます。5 は 4 の `production/features` も読みます。
- 6〜12 はすべて 5（`key-query-01/production/`）を読みます。
- 6 の `lora_theta_on_selected.py` は 4 の LoRA 出力を読みます。7、8 の `k2_unseen_layers.py`、12 の `check` は 6 の出力を読みます。
- 11 は 9 の `key-query-03/prep.npz` を読みます。12 で単位メモリを使うときは 11 の出力を指定します。
- 1〜2 の前に段階 0 を済ませてください。`run-01/` だけで約 92 GB を使います。

段階 0 と、それより前のデノイズ前の比較（[README](README.md) の上半分の文書）は、各文書の末尾に実行コマンドがあります。

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
python $UL/prepare_oracle.py prepare
python $UL/prepare_oracle.py extract --batch 4
python $UL/oracle_experiment.py drafts
python $UL/oracle_experiment.py train-queue --max-parallel 2
python $UL/oracle_experiment.py evaluate-all
python $UL/oracle_experiment.py finalize
```

`prepare_oracle.py` は `oracle-01/unseen_rows/` に Dev と Eval の窓、VLM の特徴を書きます。
`oracle_experiment.py` は、正解で選んだ下書きを Expert で戻す17条件（[表現上限のオフライン比較](oracle_results.md)）です。
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
環境変数 `MEM_EXPERT` に [圧縮表現から始めるデノイズ](compressed_denoise.md) の Expert を指定すると、公開 Expert の代わりに使います。

**集計の注意：** LIBERO-10 の出力に `report --suite libero_10` を使うと、0/0 と表示されます。基準との対応づけが LIBERO-90 用のためです。LIBERO-10 の成功数は `memory/attempts/*/attempt-*/result.json` の `success` を数えてください。

## 確認のしかた

各段の結果は、出力の `report.md`（または `summary.json`）に書かれます。研究ノートの表と比べてください。
学習の乱数とGPUの違いで小数第3位は揺れます。結論（どれが基準より良いか、どれが差の何割を埋めるか）が変わるかで判断してください。

## テスト

データや重みなしで、合成入力で確かめるテストです。

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_unseen_*.py
```

保存済みの出力がない環境では、出力に依存する一部のテストは skip になります。
