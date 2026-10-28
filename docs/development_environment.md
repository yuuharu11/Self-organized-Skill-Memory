# 現在のコンテナでの開発環境

すでに起動しているコンテナ内で開発・実験します。追加のコンテナ構築・起動ファイルは管理しません。
SkillNetの実行ルートは `skill_moe/skillnet`、現在のリポジトリルートは `/work` です。

## 現在の環境を使う

リポジトリのルートで実行します。

```bash
source env/activate.sh
python env/smoke_test.py --render
python -m pytest -q tests/test_websocket_policy_server.py
```

`env/activate.sh` は `SKILLNET_VENV`、アクティブなvenv、リポジトリの `.venv` の順で環境を選びます。
いずれも未指定・未作成の場合は、このセッションで検証済みの `/tmp/skillnet-validation` を使います。
LIBEROの完全ソースはインストール情報から検出し、必要なら `SKILLNET_LIBERO_SOURCE` で指定します。
現在の検証用共有ライブラリも一時領域にあるため、再起動後の永続性は保証しません。
別のEGLライブラリ配置を使う場合は `SKILLNET_EGL_LIBRARY_DIR` を指定できます。
NVIDIAのEGL実装は `env/egl_nvidia.json` を通じて使用します。

このスクリプトは依存パッケージや重みをダウンロードしません。LIBEROのパス設定とキャッシュ用ディレクトリは作成します。
固定したLeRobotは `LEROBOT_HOME` を拒否するので、`HF_LEROBOT_HOME` を使用してください。

## 依存関係と保存先

`env/requirements.in` が直接依存、`env/requirements.lock` がPython 3.10/Linux x86_64用の固定リストです。
JAX 0.5.3、Flax 0.10.2、PyTorch 2.7.1、MuJoCo 2.3.7、Robosuite 1.4.1を含みます。
ロックは以前の検証から内容を変更せず移動しています。

新しいvenvを用意する場合のPython依存のインストールは以下です。uv 0.8.22を使用します。

```bash
uv venv --python 3.10 .venv
cd env
uv pip sync --python ../.venv/bin/python requirements.lock
cd ..
```

これはPython依存のみの手順です。LIBEROの配布wheelだけでは完全ソース・アセットを利用できません。
固定リビジョン `8f1084e3132a39270c3a13ebe37270a43ece2a01` の完全ソースを別途配置し、
`SKILLNET_LIBERO_SOURCE` にそのルートを指定してください。
元の環境で使ったソースはLIBEROのeditableインストール情報から確認できます。
アセット登録は以下のヘルパーで行います。

```bash
python skill_moe/skillnet/examples/libero/install_libero_skill_assets.py \
  --libero-root "$SKILLNET_LIBERO_SOURCE/libero/libero" --install
```

再構築にはCMake・コンパイラ、GLVND/EGL・GLib・OpenGL等の共有ライブラリも必要です。
現在検証済みの環境を移動するだけで新しいvenvになるとは扱いません。

データ・重み・キャッシュ・実験結果は `.skillnet-state/` に保持します。
別の保存先には `SKILLNET_STATE_DIR` を指定できます。
主なサブディレクトリは `data/`、`checkpoints/`、`cache/`、`outputs/`、`validation/` です。

## 公開重みの評価

取得済みのLIBERO-40重みはそのまま使えます。

```bash
bash env/eval_smoke.sh
```

Spatialの10タスクを各1試行で評価し、保存先の `outputs/eval-smoke-<実行ID>/` にログと動画を残します。
既存の評価結果は上書きしません。`SKILLNET_RUN_DIR` で出力先を指定できます。
完全な論文評価の代わりにはなりません。

別途重みを取得する場合は以下です。ダウンロードは明示したときだけ行います。

```bash
source env/activate.sh
python env/download_checkpoint.py 40
# LIBERO-90は引数90。保存先変更は --checkpoint-root を指定。
```

検証記録は [environment_validation.md](environment_validation.md)、
研究案は [skill_representation_review.md](research/skill_representation_review.md) を参照してください。
