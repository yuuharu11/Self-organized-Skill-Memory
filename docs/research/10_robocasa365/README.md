# RoboCasa365：atomicからComposite-Unseenへ

**問い：** atomic のデモから作ったスキルメモリが Composite-Unseen のゼロショット汎化を上げるか。

**現在地：** 環境固定・task registry・評価クライアント・再生経路を実装済み。実環境の描画、smokeデータ取得、再生評価は個別に検証し、未確認項目を手順書に残す。本番学習・全タスク評価は未実行で、性能上の結論はない。

## ファイル

- [実験計画](plan.md)：分割、対照、容量・計算見積もり
- [環境と配線手順](../../robocasa365.md)：cloneからのセットアップ

## 固定プロトコル

- 学習データとメモリ構築：target split の atomic_seen 18タスクのみ。
- 主評価：target split の composite_unseen 16タスク×50試行。
- 補助評価：atomic_seen、composite_seen。
- 対照：pi05_base を target atomic_seen で微調整した方策、利用可能な公式報告値（条件差を併記）、raw区間記憶。
- 評価対象タスクの正解行動は採点だけに使い、学習・メモリ作成・検索器の調整に使わない。

## 準備で分かったこと

- **sudo は不要。** 描画は NVIDIA ドライバの EGL（`env/egl_nvidia.json`）で通り、OSMesa もシステムの FFmpeg も要らない。動画は PyAV 同梱の FFmpeg で読み、π0.5 の公開重みは認証なしで取得できる。前提は `env/check_host.sh` で確かめる。
- **正規化の落とし穴。** atomic では移動台と制御モードの行動が 98% 以上のフレームで一定で、分位点の幅が 0 になる。そのまま分位点で正規化すると損失が 2.4e11 になった。幅 0.01 未満の次元だけ分位点を両側に 1 広げ、統計の保存時に適用して学習と推論で一致させた（損失 0.118）。
- **方策は通常の π0.5。** スキル ID・MoE の Expert は使わない（LIBERO の現行実験と同じモデル）。
- **初回推論の JIT。** π0.5 の最初の推論は数分かかり、websocket の keepalive で切れる。評価クライアントは ping を切って接続する。

手順と確認済みの範囲は [セットアップ手順](../../robocasa365.md) の「確認した範囲」にある。atomic 全量の取得、学習、Composite-Unseen の本評価は未着手。
