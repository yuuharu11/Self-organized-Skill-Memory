# 現行アルゴリズムの再現手順

π0.5 LIBERO-40 の主経路（生行動メモリ＋M2検索＋修正Expert）を、公開データから評価まで通す手順です。
アルゴリズムだけを別プロジェクトで使う場合は、この手順は不要です。[Skill Memory Core](../skill_memory_core/README.md) と [アルゴリズム仕様](../skill_memory_core/ALGORITHM.md) を参照してください。

> **検証の範囲：** この順序を新しい環境で最初から通した記録はありません。順序とコマンドは保存済みの成果物の依存関係と実行記録から整理したものです。データ変換は既存データの先頭2エピソードと一致することまで確認しました。

## 全体の流れ

各段の出力が次の段の入力です。上から順に実行します。

| # | 段階 | スクリプト（`pi05_memory/` 以下） | 出力（`$SKILLNET_PI05_ROOT` 以下） |
|---|---|---|---|
| 1 | データと重みの準備 | [convert_libero_unsegmented.py](../data_process/libero/convert_libero_unsegmented.py) | `$HF_LEROBOT_HOME/emergent/libero_40_unsegmented` |
| 2 | 特徴抽出・圧縮器・基準Expertの学習 | `run.py` | `run-01/` |
| 3 | 検索キーと行動距離の計算 | `score_audit/` | `score-audit-01/` |
| 4 | 下書き依存性の確認 | `draft_sensitivity/run.py` | `draft-sensitivity-01/` |
| 5 | 難しい行の特定 | `memory_gap/run.py` | `memory-gap-01/` |
| 6 | 生行動に近い候補の検索 | `score_audit/retrieval_4b.py` | `retrieval-4b-01/` |
| 7 | Long の候補診断 | `topk_long/run.py` | `topk-long-01/` |
| 8 | Long を除いたメモリの構築 | `skill_memories/build.py` | `skill-memories-01/` |
| 9 | 局所キーの評価 | `local_key/run.py` | `local-key-02/` |
| 10 | M2 検索キーの学習 | `key_mechanism/run.py` | `key-mechanism-01/` |
| 11 | 修正Expertの学習と判定 | `fair_expert/` | `fair-expert-01/` |

アルゴリズムの本体は 8、10、11 です。他の段は、それらの入力（行の分割、難しい行、比較用のキー）を作ります。

## 準備

### 環境

[再現ガイドのセットアップ](reproduction.md#2-別環境でのセットアップ) を済ませ、リポジトリのルートで次を実行します。

```bash
source env/activate.sh
export SKILLNET_PI05_ROOT="$SKILLNET_STATE_DIR/outputs/memory-pilot/pi05-memory"
export SKILLNET_LIBERO_HDF5="$SKILLNET_STATE_DIR/data/LIBERO"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PI05=skill_moe/skillnet/examples/memory_pilot/pi05_memory
```

各スクリプトの既定の保存先はこの環境変数から決まります。GPU が必要です。`run-01/` だけで約 92 GB を使います。

| 環境変数 | 内容 | 未指定のとき |
|---|---|---|
| `SKILLNET_STATE_DIR` | データ・重み・出力の保存先 | `<リポジトリ>/.skillnet-state` |
| `SKILLNET_PI05_ROOT` | この実験の出力先 | `$SKILLNET_STATE_DIR` 以下の、元の実験の run 名 |
| `SKILLNET_LIBERO_HDF5` | LIBERO の HDF5 | `<リポジトリ>/data/LIBERO` |

### 1. データと重み

公開データと公式の重みを取得します。

```bash
huggingface-cli download openvla/modified_libero_rlds --repo-type dataset \
  --local-dir "$SKILLNET_STATE_DIR/data/libero/libero40_rlds"
mkdir -p "$SKILLNET_STATE_DIR/checkpoints/openpi-assets/checkpoints"
gcloud storage cp -r gs://openpi-assets/checkpoints/pi05_libero \
  "$SKILLNET_STATE_DIR/checkpoints/openpi-assets/checkpoints/"
```

段階 7 は `$SKILLNET_LIBERO_HDF5/libero_10` の HDF5 も読みます。LIBERO 公式の `benchmark_scripts/download_libero_datasets.py` で取得してください。

固定された `tensorflow-metadata` は TFDS の読み込みで Protobuf エラーになるため、変換のときだけ互換版を優先します。

```bash
uv pip install --no-deps --python "$SKILLNET_VENV/bin/python" \
  --target "$SKILLNET_STATE_DIR/envs/tfds-overlay" tensorflow-metadata==1.16.1

PYTHONPATH="$SKILLNET_STATE_DIR/envs/tfds-overlay:$PYTHONPATH" \
  python data_process/libero/convert_libero_unsegmented.py \
  --preset libero40 \
  --data-dir "$SKILLNET_STATE_DIR/data/libero/libero40_rlds" \
  --slice-index data_process/libero/slice_indices/libero40_slice_index.json \
  --output-repo-id emergent/libero_40_unsegmented
```

完成すると 40 タスク、1,693 エピソード、273,465 フレームになります。

## 実行

### 2. 特徴抽出と基準Expert

```bash
python $PI05/run.py pipeline --output "$SKILLNET_PI05_ROOT/run-01"
```

### 3. 検索キーと行動距離

```bash
A="$SKILLNET_PI05_ROOT/score-audit-01"
python $PI05/score_audit/oracle.py
python $PI05/score_audit/make_vlm_keys.py --output "$A/vlm-full"
python $PI05/score_audit/keys.py tabular
python $PI05/score_audit/keys.py codec
python $PI05/score_audit/keys.py pi05-full
python $PI05/score_audit/learned_key.py

# 段階6は非VLMキーを nonvlm-full/keys から読みます
mkdir -p "$A/nonvlm-full" && ln -sfn ../keys "$A/nonvlm-full/keys"
```

キーの形式と資源の見積もりは [score_audit/README.md](../skill_moe/skillnet/examples/memory_pilot/pi05_memory/score_audit/README.md) にあります。この段の順序と配置は、他の段より記録が少なく、新規環境では調整が要る可能性があります。

### 4〜9. 分割とメモリ

段階 7 以外の出力先は既定値のままです。

```bash
python $PI05/draft_sensitivity/run.py
python $PI05/memory_gap/run.py
python $PI05/score_audit/retrieval_4b.py
python $PI05/topk_long/run.py --output "$SKILLNET_PI05_ROOT/topk-long-01/production"
python $PI05/skill_memories/build.py
python $PI05/local_key/run.py
```

`memory_gap/run.py` は `memory-gap-01/` に `summary.json` と同内容の `summary_corrected.json` を書き出します。段階 8 と 10 はこのファイルの閾値を読みます。

後続の段は `topk-long-01/production/` を読みます。

### 10. M2 検索キー

```bash
python $PI05/key_mechanism/run.py --threads 16
```

後続の段は `key-mechanism-01/production/` を読みます。

### 11. 修正Expertの学習と判定

同じ初期値と更新回数で、M2 の下書きとゼロの下書きの Expert をそれぞれ学習して比べます。

```bash
python $PI05/fair_expert/run.py prepare
python $PI05/fair_expert/run.py train --expert em2 --draft-source m2
python $PI05/fair_expert/run.py train --expert zero --draft-source zero
python $PI05/fair_expert/reselect.py
python $PI05/fair_expert/run.py evaluate
python $PI05/fair_expert/review_top1_shuffle.py
```

判定基準は [fair_expert/plan.md](research/07_pi05_memory40/fair_expert/plan.md) で事前に固定しています。

## 結果の確認

元の実験の値です。保存済みの実験は 1 seed で、値は `fair-expert-01/` と `key-mechanism-01/production/` から確認できます。

| 項目 | 全 Long | 難しい行 |
|---|---:|---:|
| M2 Top-1 の生行動 MSE | .105 | .156 |
| 行動 oracle（同じメモリでの上限） | .0290 | .0351 |
| F1：E-M2＋M2 Top-1 − E-zero＋zero | −.00157 [−.00253, −.00069] | −.01050 [−.01670, −.00423] |
| F2：E-M2＋M2 Top-1 − E-M2＋シャッフル | +.000026 [−.000694, +.000780] | +.000197 [−.005805, +.007100] |

F1 は成立、F2 は不成立です。つまり「メモリで学習した Expert は有用」とは言えますが、「検索した動きの中身が効いた」とは言えません。

実装が同じであることは [PARITY.md](../skill_memory_core/PARITY.md) のテストで確認できます。GPU が必要で、JAX を CPU に固定すると失敗します。

```bash
SKILL_MEMORY_ARTIFACTS="$SKILLNET_PI05_ROOT" python -m pytest -q -s skill_memory_core/tests/test_parity.py
```

## 既知の限界

- **閉ループは未検証です。** Core 経由の閉ループ評価はまだなく、実行時の検索サーバーも未実装です。
- **保存値との差が残っています。** 生成結果は保存済みの予測と最大 .00185 ずれており、原因は特定できていません。
- **`_history` に既知の不具合があります。** 既存の重みとの互換性のため `as_trained` として残しています（[ALGORITHM.md](../skill_memory_core/ALGORITHM.md)）。
- **学習済みの M2 と Expert は公開していません。** 手順どおりに再学習が必要で、同じ値になる保証はありません。
- **Long は未知タスクではありません。** 公開 π0.5 のモデルと VLM は Long タスクを事前学習で見ています。
- 完了した過去の実験のコードは Git タグ `archive/before-cleanup` にあります。
