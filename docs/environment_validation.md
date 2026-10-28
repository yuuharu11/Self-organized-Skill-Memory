# 開発環境の検証記録

検証日: 2026-10-25（日本時間）。
対象ソース: `52e3caf52bb183413bbdda83c359f76bb41250eb` に今回の環境・互換性修正を適用した作業ツリー。
現在のコンテナで使う手順は [development_environment.md](development_environment.md) を参照してください。

## 検証した環境

現在の既存コンテナ内に隔離した `/tmp/skillnet-validation` のPython環境を作成しました。
この記録は既存コンテナ内での実測結果です。環境管理ツールは `env/` に整理しています。
この一時venvと描画用の一時共有ライブラリは再起動後の永続性を保証しません。

| 項目 | 実測・確認内容 |
| --- | --- |
| OS / Python | Ubuntu 22.04 / Python 3.10.12 |
| GPU | NVIDIA H200 NVL ×1、143771 MiB |
| ドライバ | 595.45.04（`nvidia-smi` の対応CUDA表示13.2） |
| JAX / PyTorch | 0.5.3 CUDA12 / 2.7.1 |
| 固定依存数 | 256パッケージ、`uv pip check` 成功 |
| ロックSHA256 | `6c86f06eea058738feda230dfef8a0431a62caa901edb437dcb4b3a362c5878e` |

固定したPython依存は `env/requirements.lock` に保存しています。
検証時のPython環境と一部共有ライブラリは一時領域にあり、再起動後は再配置が必要な場合があります。

## 成功した確認

- 開始時のGit作業ツリーはクリーン。公式リモート、必要ドキュメント、Gitオブジェクト整合性を確認。
- 今回の変更を含むソースコピーで、公式 `check_public_release.py --verbose` が成功。
- JAXでH200を検出し、行列積を実行。PyTorchでもCUDA上の演算を実行。
- `skillnet_client.websocket_client_policy`、`openpi.models.pi0_moe_skill`、
  `openpi.policies.policy_config`、`openpi.training.data_loader_skill` のimport成功。
- `pi05_libero_moe_skill_4_40` と `pi05_libero_moe_skill_4_90` の設定取得成功。
- LIBERO-Skillの9タスクが公開manifestと同順で登録され、タスク定義・初期状態を参照できることを確認。
- 外部LIBEROへの登録修正は、新規適用と再適用時の冪等性を確認。
- WebSocketの要求と応答がずれる重複送信を修正。
  `tests/test_websocket_policy_server.py` で8回の連続推論要求がそれぞれ自身の行動を受信する回帰テストに成功。
- NVIDIA EGLで環境を作成し、初期状態設定・reset・行動step・128×128 RGB画像取得に成功。

既存コンテナはNVIDIAのEGL実装を備えていましたが、GLVNDローダー、GLib等が不足していました。
ホストのシステム領域へインストールせず、一時ディレクトリに共有ライブラリを展開し、
NVIDIAのEGL ICDを指定して描画を検証しました。
NVIDIA ICDは `env/egl_nvidia.json` に、起動時の環境設定は `env/activate.sh` に整理しています。

## 公開チェックポイント評価

`jsw19/SkillNet-LIBERO-40` の取得が完了しました。
Hub revisionは `2505898fbf65d06994711ca69a2141f904c9368f`、ファイル合計は約15.8 GBです。
保存先は以下で、現在の環境からそのまま再利用できます。

```text
.skillnet-state/checkpoints/pi05_libero_moe_skill_4_40/moe_add_balance_4_128_30000/29999/
```

`libero_spatial`、全10タスク、各1試行、seed 7、`ZERO_SHOT=0` の評価スモークが
**10/10成功（100%）、終了コード0**で完了しました。
実行時刻は日本時間17:04〜17:07、ロールアウト部分は約2分27秒でした。
これは小規模な動作確認であり、全4スイート・各50試行の論文再現を意味しません。

公開重みの復元、ポリシーサーバー起動、クライアント通信、NVIDIA EGL描画、
行動生成・実行、成功判定、全10本の動画保存を確認しました。
評価中のGPUメモリ表示は約17960 MiBでした（定点観測であり、最大値や最低必要容量の測定ではありません）。
終了後はサーバーが停止し、GPUメモリ使用量が0 MiBへ戻ったことも確認しました。

初回試行ではWebSocketサーバーの重複応答が見つかり、6タスク目で通信タイムアウトになりました。
1要求1応答への修正と通信回帰テストを行った後の上記結果を採用します。
修正前ログは `evaluation-before-protocol-fix.log` と `server-before-protocol-fix.log` に保持しています。
修正後動画は `.skillnet-state/outputs/eval-smoke/videos-after-fix/` にあります。

## ログと保存先

| 内容 | 保存先 |
| --- | --- |
| 公式軽量チェック | `.skillnet-state/validation/public-release.log` |
| 依存整合性 | `.skillnet-state/validation/pip-check.log` |
| import / GPU演算 | `.skillnet-state/validation/import-gpu-smoke.log` |
| NVIDIA EGL / シミュレータ | `.skillnet-state/validation/egl-smoke.log` |
| 通信回帰テスト | `.skillnet-state/validation/websocket-test.log` |
| 公開重み取得 | `.skillnet-state/validation/checkpoint-download.log` |
| 検証条件の機械可読記録 | `.skillnet-state/validation/manifest.json` |
| 評価・サーバーログ・動画 | `.skillnet-state/outputs/eval-smoke/` |

大容量データを再帰走査しないよう、公式軽量チェックはローカルのソースのみをコピーした一時ディレクトリで実行します。
Git情報を必要とする検査には、チェック元のGitメタデータを参照します。
ログ・重み・キャッシュ・実験出力はGit対象外です。

## 残る確認

- コンテナ再起動後のPython環境・LIBERO完全ソース・OS共有ライブラリの復元は未検証。
- 完全な学習は未実施。必要GPU数、バッチ上限、学習時間は未測定。
- LIBERO-40/90の派生LeRobotデータへのアクセス可否・ローカル再構築は未検証。
  既存の公開リリース記録では派生データの匿名アクセスが未解決とされています。
- 本評価の4スイート×50試行、LIBERO-Skillの9タスク×50試行、RoboTwin評価は未実施。
- 研究モデル、DMP、メモリは未実装。SkillMemoとの差分は要旨確認までです。
