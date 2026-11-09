# RoboCasa365 の準備と配線確認

RoboCasa365 で実験するための環境構築・データ取得・評価の配線確認の手順です。**sudo（管理者権限）は使いません。** 研究上の問いと計画は [研究ノート 10](research/10_robocasa365/README.md) にあります。

## 全体像：2つの Python 環境

RoboCasa 1.0.1 は Python 3.11・MuJoCo 3.3.1・NumPy 2.2 を要求します。既存の SkillNet 環境（Python 3.10・robosuite 1.4.1・MuJoCo 2.3.7・NumPy 1.26）とは同じ環境に入りません。そこで役割を分けます。

| 環境 | 場所（既定） | 入れるもの | 使う段 |
|---|---|---|---|
| **RoboCasa 環境**（Python 3.11） | `$SKILLNET_STATE_DIR/envs/robocasa` | robocasa・robosuite（固定コミット）、通信クライアント | シミュレータ、描画、データ取得、評価クライアント |
| **SkillNet 環境**（Python 3.10） | `$SKILLNET_VENV` | openpi（π0.5）、LeRobot、JAX | 正規化統計、学習、方策サーバー、再生サーバー |

評価のとき、方策は SkillNet 環境で websocket サーバーとして動き、RoboCasa 環境のクライアントが接続します。LIBERO の評価と同じつなぎ方です。以下、各コマンドの見出しに **[RoboCasa]** / **[SkillNet]** を付けます。

固定している上流の版：RoboCasa `3443cdf3dd20f9ae8ed41adba442059b506891d0`、robosuite `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`。

## 1. 前提を確かめる（sudo なし）

clone したリポジトリのルートで、保存先を決めて確認スクリプトを実行します。何もインストールせず、足りないものごとに **sudo を使わない入れ方** を表示します。

```bash
git clone https://github.com/yuuharu11/Self-organized-Skill-Memory.git
cd Self-organized-Skill-Memory
export SKILLNET_STATE_DIR="$PWD/.skillnet-state"   # 容量のある場所なら別でもよい
export PATH="$HOME/.local/bin:$PATH"
bash env/check_host.sh
```

| 項目 | 必要な理由 | sudo なしでの入れ方 |
|---|---|---|
| uv 0.8.22 | 両環境の作成 | `curl -LsSf https://astral.sh/uv/0.8.22/install.sh \| sh` |
| Python 3.10 / 3.11 | 各環境 | `uv python install 3.10`（3.11 は setup が自動で入れる） |
| CMake 3 系 | SkillNet 環境の `egl-probe` のビルド | `uv tool install "cmake<4"` |
| gcc・g++・make | 同上 | 通常は OS に入っている。無ければ管理者に依頼するか、conda/micromamba のコンパイラをホームに入れる |
| NVIDIA ドライバ、`libGL.so.1`・`libEGL.so.1`・`libEGL_nvidia.so.0` | GPU とヘッドレス描画 | GPU マシンなら通常入っている。無ければ管理者に依頼 |

- **描画**：NVIDIA ドライバの EGL を使います。`env/robocasa/activate.sh` が `__EGL_VENDOR_LIBRARY_FILENAMES` を [env/egl_nvidia.json](../env/egl_nvidia.json) に向けるので、`/dev/dri` へのアクセス権も root も要りません。NVIDIA EGL が無いホストでは、`libOSMesa` があれば `export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` で CPU 描画に切り替えられます（遅い）。
- **FFmpeg はシステムに不要です。** 学習データの動画は、PyAV の wheel に同梱された FFmpeg で読みます（`SKILLNET_ROBOCASA_VIDEO_BACKEND`、既定 `pyav`）。評価動画の書き出しには `imageio-ffmpeg` の同梱バイナリを使います。
- **gcloud も不要です。** π0.5 の公開重み（`gs://openpi-assets`）は認証なしで読めるので、初回使用時に自動で取得します。

## 2. 環境を作る

### SkillNet 環境

既存の手順（[再現ガイド](reproduction.md#2-別環境でのセットアップ)）と同じです。`apt-get` の段は飛ばし、1. の確認が通っていれば次だけで作れます。

```bash
export SKILLNET_VENV="$SKILLNET_STATE_DIR/envs/skillnet"
bash env/setup.sh
```

### RoboCasa 環境

```bash
bash env/robocasa/setup.sh --assets
```

Python 3.11 の venv を作り、robocasa と robosuite を固定コミットで `$SKILLNET_STATE_DIR/sources/` に clone して入れます。マクロを設定し、データの保存先（`DATASET_BASE_PATH`）を `$SKILLNET_STATE_DIR/data/robocasa` に向けます。`--assets` はキッチンのアセット（約10GB）を取得します。完了の印があれば再取得はしません。依存は [env/robocasa/requirements.lock](../env/robocasa/requirements.lock) に固定しています。checkout にローカルの変更があれば、版の切り替えをせずに止まります。

## 3. 描画のスモーク [RoboCasa]

```bash
source env/robocasa/activate.sh
python env/robocasa/smoke_test.py --render --output "$SKILLNET_STATE_DIR/validation/robocasa-render"
```

target split の `OpenDrawer`（atomic）と `ArrangeBreadBasket`（composite_unseen）を開き、2ステップ動かして、3カメラ（左右の agentview と手首）の PNG を保存します。タスク集合の件数（atomic_seen 18、composite_seen 16、composite_unseen 16）も確かめます。

タスク集合は、固定コミットの registry から [data_process/robocasa/task_sets.json](../data_process/robocasa/task_sets.json) に書き出して Git に入れています（setup が再生成します）。SkillNet 環境は robocasa を import できないので、この JSON だけを読みます。

## 4. 動作確認用データ [RoboCasa]

```bash
python data_process/robocasa/download.py --preset smoke --split target --source human
python data_process/robocasa/inspect_dataset.py
```

smoke では target human の `OpenDrawer` 1タスクだけを取得します（514 エピソード、20 Hz）。取得記録は `$SKILLNET_STATE_DIR/data/robocasa/download_manifest.json` に残ります。

データは LeRobot 形式です（`meta/`、`data/chunk-*/` の parquet、3カメラの `videos/`、`extras/` にエピソードごとの `ep_meta.json`・`model.xml.gz`・`states.npz`）。

| 行動（12次元） | 保存データの順 | 方策・環境の順（`convert_action`） |
|---|---|---|
| 移動台 | 0–3 | 7–10 |
| 制御モード | 4 | 11 |
| 手先の位置 | 5–7 | 0–2 |
| 手先の回転 | 8–10 | 3–5 |
| グリッパ | 11 | 6 |

学習時は `RobocasaInputs` が保存データの順から方策の順に並べ替えます。出力は方策の順のまま環境に渡します。状態は16次元で、移動台の位置3、移動台の回転（四元数）4、手先の相対位置3、手先の相対回転（四元数）4、グリッパの関節2です。

**正規化の注意：** atomic のデータでは、移動台と制御モードの行動が 98% 以上のフレームで一定です。そのため分位点の幅が 0 になり、まれな値が 1e-6 で割られて巨大になります（損失が 1e11 に達した）。RoboCasa の設定では、幅が 0.01 未満の次元だけ分位点を両側に 1 広げます（`_widen_degenerate_quantiles`）。この処理は統計の保存時に入るので、学習と推論で同じ値を使います。

## 5. 再生による配線確認（π0.5 なし）

```bash
source env/robocasa/activate.sh            # クライアントは RoboCasa 環境
bash env/robocasa/eval_smoke.sh            # サーバーは $SKILLNET_VENV の Python で自動起動
```

再生サーバー（SkillNet 環境）が、smoke データのエピソード 0 の行動をそのまま返します。クライアントは同じ `model.xml.gz` と `states.npz` から初期状態を復元し、OpenDrawer を1試行実行します。**成功判定が true になれば**、クライアント、行動の並べ替え、環境の成功判定がつながっています。方策の性能の評価ではありません。空きポートは自動で選びます（`ROBOCASA_SMOKE_PORT` で指定も可）。結果は `$SKILLNET_STATE_DIR/outputs/robocasa-eval-smoke-<id>/results.json` に、サーバーのログは `$SKILLNET_STATE_DIR/logs/` に残ります。

## 6. π0.5 での配線確認 [SkillNet] → [RoboCasa]

学習を始める前に、正規化統計、モデル、方策サーバー、クライアントが通ることを確かめます。smoke データは1タスクだけなので、部分集合を許す環境変数を明示します（本番では外す。外すと18タスクが揃うまで止まる）。

```bash
# [SkillNet] 正規化統計（smoke 分）、モデルの forward、サーバー用のチェックポイント
source env/activate.sh
export SKILLNET_ROBOCASA_ALLOW_PARTIAL=1
python skill_moe/skillnet/scripts/compute_norm_stats_moe_skill.py --config-name pi05_robocasa_atomic --max-frames 4096
python skill_moe/skillnet/examples/robocasa/model_smoke.py
CKPT=$(python skill_moe/skillnet/examples/robocasa/link_base_checkpoint.py | tail -1)
python skill_moe/skillnet/scripts/serve_policy_moe_skill.py --port 8000 \
  policy:checkpoint --policy.config pi05_robocasa_atomic --policy.dir "$CKPT"
```

```bash
# [RoboCasa] 別の端末で
source env/robocasa/activate.sh
python skill_moe/skillnet/examples/robocasa/main.py --tasks OpenDrawer --split target --num-trials 1 \
  --port 8000 --output "$SKILLNET_STATE_DIR/outputs/robocasa-eval-wiring-$(date -u +%Y%m%dT%H%M%SZ)"
```

- `model_smoke.py` は1バッチ読み込み、pi05_base の重みで**通常の π0.5**（Expert は gemma_300m、スキル ID・MoE なし）を作り、損失を1回計算します。最後に `MODEL SMOKE OK` と表示されれば成功です。
- `link_base_checkpoint.py` は、pi05_base の重みと RoboCasa の正規化統計をシンボリックリンクで並べたディレクトリを作ります。pi05_base は RoboCasa で微調整していないので、**この成功率は比較の基準ではありません**。確認するのは、サーバーが起動し、クライアントが最後まで回ることだけです。
- 本番の評価では `--policy.dir` を微調整後のチェックポイントにし、`--task-set composite_unseen --num-trials 50` を指定します。`get_eval_stats.py <results.json>` でタスク別の成功率を集計します。

## 7. 本番（未実行）

ここから先は大きな資源が要るため、まだ実行していません。

```bash
# [RoboCasa] target の atomic_seen 18タスクを取得
python data_process/robocasa/download.py --task-set atomic_seen --split target --source human
# [SkillNet] 全量の正規化統計（ALLOW_PARTIAL と --max-frames を外す）→ 微調整
unset SKILLNET_ROBOCASA_ALLOW_PARTIAL
python skill_moe/skillnet/scripts/compute_norm_stats_moe_skill.py --config-name pi05_robocasa_atomic
python skill_moe/skillnet/scripts/train_moe_skill.py pi05_robocasa_atomic --exp-name <名前>
```

- 容量は動画の圧縮率しだいで、実測していません。smoke の OpenDrawer 1タスクの大きさを `du -sh` で測り、18倍を目安にしてください。
- 微調整の設定（`pi05_robocasa_atomic`）は、pi05_base から始め、行動の長さ 10、バッチ 32、30k 更新です。学習率と更新数は研究計画で決めてから実行してください。
- 単位の切り出し（メモリ側）：`python skill_moe/skillnet/examples/memory_pilot/robocasa/units.py <データのパス>`。速度の極小点とグリッパの切り替えで切り、手先6次元とグリッパを型の特徴にします。移動台と制御モードは別のチャネルとして持ち、既定では切れ目に使いません。タスク名やサブタスク、成功のラベルは使いません。

## 8. 確認した範囲

| 段 | 結果 |
|---|---|
| `env/check_host.sh` | root で sudo の無いコンテナで、CMake 以外を検出。CMake の案内を表示 |
| `env/robocasa/setup.sh --assets` | Python 3.11 の venv、固定コミット、アセット取得まで完了 |
| `smoke_test.py --render` | NVIDIA EGL（`egl_nvidia.json`）で3カメラの PNG を保存。OSMesa は不要 |
| `download.py --preset smoke`・`inspect_dataset.py` | OpenDrawer 514 エピソード、20 Hz、行動 12、状態 16 |
| `eval_smoke.sh`（再生） | OpenDrawer で成功判定 true（318 ステップ） |
| `model_smoke.py`（通常の π0.5） | 1バッチの損失 0.118（正規化の保護を入れる前は 2.4e11） |
| π0.5 サーバー＋`main.py`（pi05_base、未微調整） | サーバー起動、正規化統計の読込、OpenDrawer を上限の 750 ステップまで実行し動画と `results.json` を保存（成功 0/1。未微調整なので想定どおり） |
| `units.py` | 514 エピソードから 7,505 単位、長さの中央値 16 ステップ |

**新しく clone した状態からの通し確認：** 別のディレクトリに clone し、新しい `SKILLNET_STATE_DIR`（既存の環境・データ・重みを使わない）で 1.〜6. とテストを上から順に実行して、すべて通過した。前提の確認で足りなかったのは CMake だけで、表示された `uv tool install "cmake<4"` で入れた。各段の所要時間は、前提の確認から SkillNet 環境まで約2分、RoboCasa 環境とアセットで約20分、描画・データ・再生で約3分、π0.5 の確認（重み 12GB を認証なしで取得、正規化統計、モデル、サーバーとクライアント）で約41分。再生の成功判定は true、π0.5 の損失は 0.119。

微調整、全量の統計、Composite-Unseen 16タスク×50試行の評価は未実行です。RoboCasa のアセットとデータの利用条件は、利用者が確認してください。
