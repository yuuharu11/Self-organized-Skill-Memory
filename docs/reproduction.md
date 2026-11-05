# 別環境での再現ガイド

この文書は、リポジトリを別の PC・サーバーへ渡して実行するための入口です。
対象は取得したコミットに含まれるコードです。研究の説明は
[研究ノート](research/README.md)、元のコンテナの実測記録は
[環境検証記録](environment_validation.md) を参照してください。

## 1. 再現する対象を選ぶ

| 対象 | 入力・実行条件 | 確認できるもの |
| --- | --- | --- |
| SkillNet の階層トークン | Git の小さな公開ファイル、Python | 公開済みトークナイザの出力。新しいスキルメモリとは別の仕組み |
| 研究コードの単体確認 | Python 依存、合成入力 | スキル表現・検索・修正 Expert の実装。データ・重み・GPU は不要 |
| 公開 SkillNet-LIBERO-40 の評価スモーク | 公開重み、LIBERO 完全ソースとアセット、GPU、描画環境 | 公開方策の読込から環境実行まで |
| 現行アルゴリズムの通し実行 | 全エピソードを保持した LIBERO-40、π0.5 重み、GPU | データ変換から修正 Expert の判定まで。[再現手順](reproduce_core_path.md) |

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
  tests/test_skill_methods_m1.py tests/test_pi05_memory_core.py \
  tests/test_pi05_fair_expert.py tests/test_pi05_key_mechanism.py \
  skill_memory_core/tests/test_core.py
```

トークナイザの例では `tokens: [1, 7, 57]` が出力されます。
続くテストの成功は終了コード0で確認します。各テストは合成入力を使います。
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

## 4. 研究実験の入力と実行

現行アルゴリズムの入力は、**元のデモ1本を LeRobot の1エピソードとして保持した LIBERO-40** です（40タスク・1,693エピソード・273,465フレーム）。
タスク番号は `0–9 = LIBERO-10`、`10–19 = goal`、`20–29 = object`、`30–39 = spatial` です。
既存の公開変換器が作る `libero_40_v1` はスキル区間ごとに分割した3,862エピソードの形式で、代用すると分割と評価条件が変わります。

データの作成と実行順序は [現行アルゴリズムの再現手順](reproduce_core_path.md) にあります。
公開 RLDS の取得と、ベースの SkillNet 用の変換手順は [データ処理ガイド](libero_data_processing.md) にあります。
完了した過去の実験のコードは Git タグ `archive/before-cleanup` にあります。

## 5. 検証の範囲と既知の問題

この案内の追加時点では、元のコンテナ上で CLI・小規模テスト・リンクを確認しています。
新規 GPU マシンでの環境構築やデータの再抽出、長時間の学習・全試行の評価を完了したという記録ではありません。

| 症状 | 確認する箇所 |
| --- | --- |
| `source env/activate.sh` で環境がない | `.venv` を作成し、`SKILLNET_VENV` を指定する。スクリプト自体は依存を導入しない |
| LIBERO のアセット・benchmark が見つからない | 完全ソースの固定リビジョン、editable install、アセット登録を確認する |
| `libGL.so.1` がない・EGL 初期化失敗 | OS の OpenGL/EGL ライブラリ、NVIDIA ドライバ、コンテナの GPU 共有を確認する |
| TFDS import 時の Protobuf エラー | 固定リストの `tensorflow-metadata==1.21.0` と Protobuf 5.29.6 の生成コード互換性に問題がある。データ変換前に import を確認し、互換環境を別途検証する。元の変換処理はローカルの互換 overlay を使用していた |


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
