# 別環境での再現ガイド

この文書は、リポジトリを別の PC・サーバーへ渡して実行するための入口です。
対象は取得したコミットに含まれるコードです。研究の説明は
[研究ノート](research/README.md)、元のコンテナの実測記録は
[環境検証記録](environment_validation.md) を参照してください。

## 1. 再現する対象を選ぶ

| 対象 | 入力・実行条件 | 確認できるもの |
| --- | --- | --- |
| SkillNet の階層トークン | Git の小さな公開ファイル、Python | 公開済みトークナイザの出力。新しいスキルメモリとは別の仕組み |
| 軽量研究コードの単体確認 | Python 依存、合成入力 | 入力変換・検索・デノイズ等の実装。データ・重み・GPU は不要 |
| 公開 SkillNet-LIBERO-40 の評価スモーク | 公開重み、LIBERO 完全ソースとアセット、GPU、描画環境 | 公開方策の読込から環境実行まで |
| 初期の軽量スキル実験 | 全エピソードを保持した LIBERO-40 LeRobot データ、π0.5 base 重み | 凍結 VLM 特徴の抽出と小さなモデルのオフライン評価 |
| S1/S2 の研究ノートの数値比較 | 対応するデータ・特徴キャッシュ・学習済みメモリ・設定・コード版 | 後述の追加資産を揃えた条件での比較 |

公開重みの評価スモークは、スキルメモリの改善や論文の成功率を再現する実験とは別に報告してください。
元のコンテナでは Spatial の各1試行が10/10成功しましたが、各50試行の論文評価ではありません。
必要 VRAM の下限・別 GPU での速度は測定していません。

## 2. 別環境でのセットアップ

### コード版と保存先

[README の取得手順](../README.md#コードを取得する) で対象のコード版を選び、リポジトリのルートで実行します。
レビュー中のPRは README のPR取得手順で選択できます。mainに取り込まれた内容は通常のcloneで取得できます。

```bash
export SKILLNET_REPO_ROOT="$(pwd)"
export SKILLNET_STATE_DIR="$SKILLNET_REPO_ROOT/.skillnet-state"
export SKILLNET_VENV="$SKILLNET_REPO_ROOT/.venv"
export SKILLNET_LIBERO_SOURCE="$SKILLNET_STATE_DIR/sources/LIBERO"
mkdir -p "$SKILLNET_STATE_DIR/validation"
git rev-parse HEAD > "$SKILLNET_STATE_DIR/validation/source_commit.txt"
```

`SKILLNET_STATE_DIR` は容量のある別ディスクのディレクトリにも変更できます。
Git はソース・設定・小さな注釈を管理します。データ・重み・キャッシュ・実行結果はこの保存先で管理します。
元のコンテナの一時 venv や既存のデータは、新しい clone にはありません。

### Python と LIBERO

固定リストの対象は **Linux x86_64 / Python 3.10** です。
事前に Python 3.10、Git、uv 0.8.22、CMake と C/C++ コンパイラを用意してください。
GPU 実験には CUDA 対応 NVIDIA ドライバが必要です。コンテナでは GPU とドライバの共有も設定します。
描画には `libGL.so.1`・`libEGL.so.1`・GLib と、ドライバの EGL 実装が必要です。
Python の依存導入だけでは OS ライブラリや GPU の設定は完了しません。

```bash
uv --version
uv venv --python 3.10 "$SKILLNET_VENV"
cd "$SKILLNET_REPO_ROOT/env"
uv pip sync --python "$SKILLNET_VENV/bin/python" requirements.lock
cd "$SKILLNET_REPO_ROOT"

mkdir -p "$SKILLNET_STATE_DIR/sources"
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$SKILLNET_LIBERO_SOURCE"
git -C "$SKILLNET_LIBERO_SOURCE" checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
uv pip install --python "$SKILLNET_VENV/bin/python" --no-deps -e "$SKILLNET_LIBERO_SOURCE"

unset LEROBOT_HOME
source env/activate.sh
python skill_moe/skillnet/examples/libero/install_libero_skill_assets.py \
  --libero-root "$SKILLNET_LIBERO_SOURCE/libero/libero" --install
uv pip check --python "$SKILLNET_VENV/bin/python"
```

ロック内の相対 editable パスを解決するため、`uv pip sync` は `env/` から実行します。
LIBERO は完全ソースを editable にして、シミュレータのアセットも参照できるようにします。
登録ヘルパーはこの LIBERO チェックアウトに公開 LIBERO-Skill のアセットを追加し、変更対象のテキストはバックアップします。
固定した LeRobot では保存先の変数名は **`HF_LEROBOT_HOME`** です。
旧 SkillNet 文書中の `LEROBOT_HOME` は、この環境では設定せず `HF_LEROBOT_HOME` と読み替えてください。

### 確認する順序

まずデータや重みをダウンロードせずに確認できます。

```bash
python data_process/skill_hierarchy/skill_hierarchy_tokenizer.py \
  --phrase "open drawer" --motion-code 200100

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_skill_methods_m4.py tests/test_skill_methods_m5.py \
  tests/test_skill_intermediate_action.py tests/test_skill_intermediate_clip.py \
  tests/test_skill_intermediate_ambiguity.py \
  tests/test_emergent_fair_compare.py tests/test_emergent_roundtrip.py
```

トークナイザの例では `tokens: [1, 7, 57]` が出力されます。
続くテストはこのPRの確認時に57件成功しました。終了コード0が、この範囲の成功の目印です。各テストは合成入力を使います。
続いて GPU と描画を個別に確認します。

```bash
python env/smoke_test.py
python env/smoke_test.py --render
```

CPU で環境の import を調べる場合は `python env/smoke_test.py --cpu` を使えますが、
このスクリプトも LIBERO の登録を確認するため、LIBERO 完全ソースが必要です。
`IMPORT OK`・`CONFIG OK` と終了コード0を確認し、描画エラーを解決してから評価へ進みます。

## 3. 公開重みで閉ループ実行する

LIBERO-40 のスモークは学習用データセットを必要としません。
環境構築後、リポジトリのルートで実行します。

```bash
python env/download_checkpoint.py 40 \
  --revision 2505898fbf65d06994711ca69a2141f904c9368f
bash env/eval_smoke.sh
```

チェックポイントには `params/` と `assets/` の両方が必要です。
取得先と Hub の確定 revision は、チェックポイント直下の `download_revision.json` に残ります。
実行ごとに `$SKILLNET_STATE_DIR/outputs/eval-smoke-<実行ID>/` が作られ、
`evaluation.log`、`server.log`、`videos/` が保存されます。
クライアントとサーバーが終了したこと、10タスク分の試行・成功判定と動画が揃ったことを確認してください。

本数を増やす場合は `NUM_TRIALS=20 bash env/eval_smoke.sh` のように明示します。
このスモークの標準設定は `libero_spatial`、各1試行、seed 7、`ZERO_SHOT=0` です。
LIBERO-Skill のゼロショット評価とは条件が異なります。
全スイートや SkillNet 自体の学習は [既存の学習・評価手順](training_and_evaluation.md) を参照してください。

## 4. 軽量研究実験の入力と実行

### 入力データの違い

軽量実験の入力は、**元のデモ1本を LeRobot の1エピソードとして保持した LIBERO-40** です。
`meta/info.json`・`meta/episodes.jsonl`・`meta/tasks.jsonl` と
`data/chunk-*/episode_*.parquet` が必要です。
測定に使った入力は 40タスク・1,693エピソード・273,465フレームでした。
タスク番号は `0–9 = LIBERO-10`、`10–19 = goal`、`20–29 = object`、`30–39 = spatial` です。
初期の軽量実験は0–9を除きます。

| 用途 | Parquet の列 |
| --- | --- |
| 凍結 VLM 特徴の抽出 | `image`, `wrist_image`, `state`, `actions`, `episode_index`, `task_index`, `frame_index` |
| 後続の中間行動実験の事後評価・行結合 | 上記に加え `eval_skill_class`, `eval_skill_segment` |

画像は Parquet に画像バイト列を保持し、状態は8次元、行動は7次元です。
`meta/episodes.jsonl` の各行には `episode_index` と `tasks` を含めます。
評価用注釈は特徴抽出器の入力列から除外されています。
Split A は object+spatial → goal、Split B は object+goal → spatial です。

**この全エピソード形式のデータセットと、その作成に使ったローカル変換ラッパーは Git に含まれていません。**
既存の公開変換器が作る `libero_40_v1` はスキル区間ごとに分割した3,862エピソードの形式です。
これをそのまま代用すると、分割・行動窓・評価条件が変わります。
同じ研究条件を再現する際は、この入力形式を満たすデータと作成時の記録も共有してください。
公開 RLDS の取得と、ベースの SkillNet 用の変換手順は [データ処理ガイド](libero_data_processing.md) にあります。

### 入力を揃えた後の初期実験

入力を `$HF_LEROBOT_HOME/emergent/libero_40_unsegmented` に配置した場合の例です。
π0.5 base の取得には Google Cloud CLI の `gcloud` を使います。
公開 SkillNet-LIBERO-40 重みと π0.5 base は用途が違うので、別の保存先を使います。

```bash
export SKILLNET_LIGHT_DATASET="$HF_LEROBOT_HOME/emergent/libero_40_unsegmented"
export SKILLNET_PI05_BASE_DIR="$SKILLNET_STATE_DIR/checkpoints/pi05_base"
mkdir -p "$SKILLNET_PI05_BASE_DIR"
gcloud storage cp -r gs://openpi-assets/checkpoints/pi05_base/params "$SKILLNET_PI05_BASE_DIR/"

test -f "$SKILLNET_LIGHT_DATASET/meta/episodes.jsonl"
test -d "$SKILLNET_PI05_BASE_DIR/params"
python -m skill_moe.skillnet.examples.emergent.light.run \
  --dataset "$SKILLNET_LIGHT_DATASET" \
  --checkpoint "$SKILLNET_PI05_BASE_DIR/params" \
  --cache-dir "$SKILLNET_STATE_DIR/cache/emergent-light" \
  --output-root "$SKILLNET_STATE_DIR/outputs/memory-pilot" \
  --vision-batch 2 --train-batch 256 --epochs 20 --seed 0
```

トークナイザは初回に取得されます。事前配置する場合は
`SKILLNET_PALIGEMMA_TOKENIZER` にローカルのモデルファイルを指定できます。
特徴抽出には π0.5 を使うため、MLP が小さくても GPU と重みの保存容量が必要です。
`--cpu` は小さなモデルの学習先を変更するオプションで、特徴抽出の重みを不要にするものではありません。
`--max-episodes` は読み出しの疎通確認用で、論文・研究ノートの比較値には使いません。

生成先は `outputs/memory-pilot/emergent-light-<日時>-<ID>/` です。
入口は `report.md`、各条件の詳細は `evaluations/H*_beta*/report.md` と `results.json`、
学習済みの小さなモデルは `models/`、特徴一覧は `feature_manifest.json` にあります。
特徴キャッシュの `manifest.json` はファイルのパスを保持するため、別のマシンへコピーした場合は
パスを更新するか、新しい環境で再抽出してください。

### S1/S2 や新しい結果も共有する場合

[研究スクリプト一覧](../skill_moe/skillnet/examples/emergent/light/intermediate/README.md) から対象を選びます。
一部の後続スクリプトには元の環境の固定パスや CLI で変更できない資産の参照が残っています。
初期実験のキャッシュを作っただけで、S1/S2 のすべてが任意の場所から実行できる状態ではありません。

研究ノートの数値まで比較する場合は、次も必要です。

- 対応するコードのコミット IDと、未コミットの変更があればその差分。
- 元のデモとタスク番号の対応、全エピソード形式の生成方法、評価用注釈。
- 特徴抽出に使った重み・トークナイザ・抽出条件と、キャッシュの manifest。
- S1 のサブタスク条件と予測テキストを使う場合の生成器・出力。S2 の LIBERO-90 メモリ・正規化統計・行分割。
- 実際のコマンド、seed、試行数、タスクごとの初期状態番号、対照条件、集計前の行単位・試行単位の結果。

10月30日の[結果・レビュー](research/reviews/experiment_results_2026-10-30.md)と、
較正・検索キー・区間スキル・場面適応・Optimus-R Stage AのコードとテストもGitで管理しています。
[S2のスクリプト一覧と必要資産](../skill_moe/skillnet/examples/memory_pilot/anchor_flow/README.md)、
[軽量実験の一覧](../skill_moe/skillnet/examples/emergent/light/intermediate/README.md)、
[Optimus-Rの実装範囲](../skill_moe/skillnet/examples/memory_pilot/optimus_r/DESIGN.md)を確認してください。
コードの配布に、元データ・大容量キャッシュ・学習済みメモリ・実験出力の配布は含まれません。
今回整理したスクリプトの既定保存先はチェックアウト先と `SKILLNET_STATE_DIR` から解決しますが、
過去のrun名を含む入力は実際に用意した資産のパスへ指定し直す必要があります。

## 5. 検証の範囲と既知の問題

この案内の追加時点では、元のコンテナ上で CLI・小規模テスト・リンクを確認しています。
新規 GPU マシンでの環境構築やデータの再抽出、長時間の学習・全試行の評価を完了したという記録ではありません。

| 症状 | 確認する箇所 |
| --- | --- |
| `source env/activate.sh` で環境がない | `.venv` を作成し、`SKILLNET_VENV` を指定する。スクリプト自体は依存を導入しない |
| LIBERO のアセット・benchmark が見つからない | 完全ソースの固定リビジョン、editable install、アセット登録を確認する |
| `libGL.so.1` がない・EGL 初期化失敗 | OS の OpenGL/EGL ライブラリ、NVIDIA ドライバ、コンテナの GPU 共有を確認する |
| TFDS import 時の Protobuf エラー | 固定リストの `tensorflow-metadata==1.21.0` と Protobuf 5.29.6 の生成コード互換性に問題がある。データ変換前に import を確認し、互換環境を別途検証する。元の変換処理はローカルの互換 overlay を使用していた |
| 広いテスト実行で失敗する | PR #2 の確認時は関連146件中140件成功、既存の z-model 4件と D3 2件が失敗。全 `tests/` は描画環境の import でも停止する |
| `check_public_release.py` が失敗する | 元の SkillNet 公開リリース契約と、この研究用リポジトリの README 等が一致せず47件失敗。PR #2 の整理前でも同じ失敗を確認済み |

`python scripts/check_public_release.py --verbose` の全成功を、この研究用チェックアウトの前提にはしていません。
個別の失敗を区別して記録してください。

## 6. 再現結果を共有する

各 run の結果とともに、以下を残すと条件を比較できます。

```bash
mkdir -p "$SKILLNET_STATE_DIR/validation"
git rev-parse HEAD > "$SKILLNET_STATE_DIR/validation/source_commit.txt"
git status --short > "$SKILLNET_STATE_DIR/validation/source_status.txt"
python --version > "$SKILLNET_STATE_DIR/validation/python_version.txt"
uv pip freeze --python "$SKILLNET_VENV/bin/python" > "$SKILLNET_STATE_DIR/validation/python_packages.txt"
sha256sum env/requirements.lock > "$SKILLNET_STATE_DIR/validation/requirements.sha256"
```

GPU 実験は `nvidia-smi` の出力も保存します。
重みの確定 revision、データの版、実行コマンド、seed と試行数、出力ディレクトリを結果に添付し、
CPU の実装確認・小規模な疎通・全条件の再実行を区別して報告してください。
