# 再現手順：RoboCasa の運動単位メモリ

**検証の範囲：** 新しい clone に今回の文書と `examples/robocasa/scripts/` を配置し、RoboCasa 関連テスト、スクリプトの構文・import・`--help` を確認します。重い学習、シミュレータ評価、GPU推論は実行しません。したがって、記載の再現コマンド全段を clone から完走した確認ではありません。環境・データのスモークは [セットアップ手順](../../robocasa365.md) を参照してください。

各段の結果は、過去の保存出力とこのノートの表を対応させています。出力先は `$R=$SKILLNET_STATE_DIR/outputs/robocasa-memory` です。実験用データ・checkpoint は別途取得します。

## 全体の流れ

| # | 段階 | スクリプト | 出力（`$R` 以下） | 保存済みの主結果 |
|---|---|---|---|---|
| 0 | 環境・データ・配線 smoke | [robocasa365.md](../../robocasa365.md) | `data/robocasa`, `envs/robocasa` | [環境手順](../../robocasa365.md) |
| 1 | 通常 π0.5 baseline | `scripts/baseline.sh` | `baseline-01/v6-nosubtask`, `target-v1`, `target18-v1` | [実験1](experiments.md#1-ベースラインと推論の確認) |
| 2 | 区間単位・P5 メモリ | `scripts/build-memory.sh` | `memory-01/policy-order-v2` | [実験2](experiments.md#2-単位メモリと-p5-キー) |
| 3 | P5 baseline、Context 特徴・キー | `scripts/context-and-p5.sh` | `ctx-01/p5_baselines`, `ctx-01/cache`, `ctx-01` | [実験2・4](experiments.md#2-単位メモリと-p5-キー), [文脈キー](experiments.md#4-文脈キー) |
| 4 | Context+history / direction rerank | `scripts/key-lite.sh` | `key-lite-01` | [実験7](experiments.md#7-キー下書き大きさの軽い比較) |
| 5 | グリッパー・大きさ head（旧・v2） | `scripts/gripper-and-magnitude.sh` | `gripper-01`, `key-lite-01/magnitude{,_v2}` | [実験5・7・9](experiments.md#5-グリッパーの決め方) |
| 6 | メモリだけの閉ループ | `scripts/memonly-closed-loop.sh` | `closed-01/memonly-01` | [実験8](experiments.md#8-メモリだけの閉ループ) |
| 7 | Expert-02 下書き準備・学習・検証 | `scripts/expert02.sh` | `expert-02/prep`, `expert-02/train` | [実験10](experiments.md#10-expert-の作り直し) |
| 8 | Expert-03 下書き・学習・t0 検証 | `scripts/expert03.sh` | `expert-03/prep`, `expert-03/train` | [実験10・11](experiments.md#10-expert-の作り直し) |
| 9 | 閉ループ：開発36、旧ヘッド再現36、全6×50、修正6×25 | `scripts/expert-closed-loop.sh` | `expert-02/closedloop-v2`, `expert-02/closedloop/attempt-01`, `expert-02/eval300`, `expert-02/fixgr` | [実験10・11](experiments.md#10-expert-の作り直し) |
| 10 | 診断：draft ablation、gap、action oracle、key diag | `scripts/draft-ablation.sh`, `scripts/diagnostics.sh` | `expert-02/draft-ablation`, `expert-02/gap`, `act-oracle`, `key-diag` | [実験10・11](experiments.md#11-ベースラインを下回る原因の切り分け) |
| 11 | 学習 reranker と閉ループ | `scripts/rerank.sh`, `scripts/rerank-closed-loop.sh` | `rerank-01` | [実験11](experiments.md#11-ベースラインを下回る原因の切り分け) |
| 12 | atomic_seen 18タスクの memory/特徴準備 | `scripts/atomic18-prepare.sh` | `memory-18` | [実験12](experiments.md#12-今の見立て) |

## 依存の要点と設定

- RoboCasa シミュレータと LeRobot データは `env/robocasa/` の Python 3.11 環境、π0.5・学習・サーバーは `SKILLNET_STATE_DIR/envs/skillnet` の既存 SkillNet 環境で動かします。データ取得、assets、ライセンス、OS/EGLの詳細は [robocasa365.md](../../robocasa365.md) に従ってください。
- checkpoint は `checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000`。実験を再実行するときは別の `SKILLNET_STATE_DIR` または新しい出力パスを使い、過去の保存物を残してください。
- `key-diag.py` の Spearman 順位相関計算には SkillNet venv の SciPy が必要です。ほかの主要な処理はリポジトリの SkillNet / RoboCasa 環境に依存します。
- baseline は `discrete_state_input=True`、action horizon 50、z-score、replan 5。SkillNet サブタスク生成は必ず `SKILLNET_SUBTASK_GENERATE=0` にします。
- 閉ループは split `pretrain`、試行seed `7 + canonical_task_index * 50 + trial`、canonical task order は CounterToCabinet, SinkToCounter, OpenDrawer, CloseFridge, SinkFaucet, TurnOffStove。Expert bridge は t0=.2、10 Euler steps。JAX は `XLA_PYTHON_CLIENT_PREALLOCATE=false` と共有GPUに合わせた `XLA_PYTHON_CLIENT_MEM_FRACTION` を指定します。
- CPU を使う Python には OMP/MKL/OpenBLAS の各 thread 数を設定してください。各 wrapper は `nice -n 10` と affinity を使います。`common.sh` は artifact path と import path の共通設定だけを行い、通信・チーム状態ファイルを読みません。

## 初期化

repository root で、書き込み可能な state directory を設定します。RoboCasa 環境の作成・データ smoke のコマンドは [robocasa365.md](../../robocasa365.md) にあります。

```bash
export SKILLNET_STATE_DIR="$HOME/.skillnet-state"
export SKILLNET_REPO_ROOT="$(pwd)"
export SKILLNET_VENV="${SKILLNET_VENV:-$SKILLNET_STATE_DIR/envs/skillnet}"
test -x "$SKILLNET_VENV/bin/python"
export SKILLNET_SUBTASK_GENERATE=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
R="$SKILLNET_STATE_DIR/outputs/robocasa-memory"
S="skill_moe/skillnet/examples/robocasa/scripts"
```

## 段ごとのコマンドと保存済みの目安

### 0–1. 環境、データ、通常方策

環境構築と1タスクの描画・replay smoke を済ませてから baseline を回します。

```bash
# docs/robocasa365.md のコードブロックを上から順に実行（Markdown ファイル自体は実行しない）
bash "$S/baseline.sh" "$R/baseline-01/v6-nosubtask" pretrain 50 12
bash "$S/baseline.sh" "$R/baseline-01/target-v1" target 50 8
bash "$S/baseline.sh" "$R/baseline-01/target18-v1" target 50 12 \
  CloseBlenderLid CloseFridge CloseToasterOvenDoor CoffeeSetupMug NavigateKitchen OpenCabinet OpenDrawer OpenStandMixerHead \
  PickPlaceCounterToCabinet PickPlaceCounterToStove PickPlaceDrawerToCounter PickPlaceSinkToCounter PickPlaceToasterToCounter \
  SlideDishwasherRack TurnOffStove TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet
```

保存値は pretrain 6×50 で **173/300**、target 6×50 で **172/300**。target 18×50 は 134/900 時点で一時停止され、`main.py --resume` 相当で再開可能です。pretrain baseline のログはサーバー起動から集計までおよそ56分、target18 の部分実行は約30分でした。

### 2–5. メモリ、文脈キー、グリッパー、大きさ

```bash
bash "$S/build-memory.sh" "$R/memory-01/policy-order-v2"
bash "$S/context-and-p5.sh" "$R/ctx-01"
bash "$S/key-lite.sh"
bash "$S/gripper-and-magnitude.sh" "$R/gripper-01"
```

`memory-01` は最初の100デモ/タスクを80/20分割し、速度極小とグリッパー切替で可変長化。Uメモリは **7,062単位**。P5の履歴あり/なしは **.0058/.0131**。Contextの top-1 型誤差は **.00581**、frame0 は **.0050**。key-lite Context+history は **.0054**、task LOTO 平均は **.0071**。グリッパー Context head の先頭5 MSEは **.096**。大きさ v2 は閉ループの1-step位置ノルム **.340**（通常π0.5 **.368**）でした。

進行ログがある段の実測壁時計：memory P5 train/evaluate は progress ログ上約1分半（先行する data/cache 作成時間は同ログにありません）；P5 baseline は約74秒；Context extract/train/evaluate は約7分；key-lite 全体は約4分16秒。gripper / magnitude v1・v2 の個別の完了時間は成果物ログに記録されていないため、数値を補いません。

### 6–9. メモリ閉ループ、Expert、主要閉ループ

```bash
bash "$S/memonly-closed-loop.sh" "$R/closed-01/memonly-reproduction"
bash "$S/expert02.sh" "$R/expert-02/prep/attempt-10" "$R/expert-02/train/attempt-04"
bash "$S/expert03.sh"
# 3-task × 12 development
bash "$S/expert-closed-loop.sh" "$R/expert-02/closedloop-v2/repro" \
  "$R/expert-02/train/attempt-04/expert/expert_001000.msgpack" 12 \
  --tasks PickPlaceCounterToCabinet OpenDrawer CloseFridge --indices 0 2 3
# Reproduce the older attempt-01 17/36 run (old magnitude head; no FIXGR flags)
MAGNITUDE_CHECKPOINT="$R/key-lite-01/magnitude/magnitude_head.pt" \
  bash "$S/expert-closed-loop.sh" "$R/expert-02/closedloop/attempt-01/repro" \
  "$R/expert-02/train/attempt-04/expert/expert_001000.msgpack" 12 \
  --tasks PickPlaceCounterToCabinet OpenDrawer CloseFridge --indices 0 2 3
# 6-task × 50 main evaluation
bash "$S/expert-closed-loop.sh" "$R/expert-02/eval300/repro" \
  "$R/expert-02/train/attempt-04/expert/expert_001000.msgpack" 50
# fix-gripper / scale-position-only evaluation: 6-task × 25
bash "$S/expert-closed-loop.sh" "$R/expert-02/fixgr/repro" \
  "$R/expert-02/train/attempt-04/expert/expert_001000.msgpack" 25 --fixgr
```

メモリだけは開発用3×12で **5/36**（通常方策17/36）。Expert-02 学習時間は保存ログで **2,362秒（39.4分）**。同じ3×12で Expert は **17/36**。6×50では **118/300** 対 baseline **173/300**。FIXGR の6×25は **47/150** 対 baseline **75/150**。Expert-03 は3,000更新、記録された学習時間 **3,622秒（60.4分）**、固定300窓の4値t0検証は **392秒（6.5分）**。6×50はログ上約2時間41分、6×25修正版は各条件と資源競合により約30分前後です。

### 10–11. 診断と reranker

```bash
bash "$S/draft-ablation.sh" "$R/expert-02/draft-ablation/repro"
bash "$S/diagnostics.sh"
bash "$S/rerank.sh"
# optional closed-loop reranker comparison
bash "$S/rerank-closed-loop.sh" "$R/rerank-01/closedloop/repro" rerank 25
```

閉ループ結果と起動条件の対応（checkpoint は Expert-02 attempt-04 の `best.json` が指す `expert_001000.msgpack`）：

| 保存済み結果 | 成績 | magnitude head | FIXGR flags | Expert 重み |
|---|---:|---|---|---|
| `expert-02/closedloop-v2/run-01` | 17/36 | `magnitude_v2/magnitude_head.pt` | なし | attempt-04 `expert_001000.msgpack` |
| `expert-02/closedloop/attempt-01` | 17/36 | 旧 `magnitude/magnitude_head.pt` | なし | attempt-04 `expert_001000.msgpack` |
| `expert-02/eval300` | 118/300 | `magnitude_v2/magnitude_head.pt` | なし | attempt-04 `expert_001000.msgpack` |
| `expert-02/fixgr` | 47/150 | `magnitude_v2/magnitude_head.pt` | `--fixgr`（draft-unit-gripper, grip-after-bridge, scale-position-only） | attempt-04 `expert_001000.msgpack` |

`expert-closed-loop.sh` は既定で FIXGR flags を付けません。attempt-01 の再現では旧 magnitude checkpoint を `MAGNITUDE_CHECKPOINT` で指定します。`closedloop-v2` と `eval300` は v2 head・flags なし、`fixgr` のみ `--fixgr` を使う対応です。

保存された診断では、保留210窓の key-top1 下書き first5 は `.72`、Expert-03 rerank-top50 は `.60`、rerank後の Expert-03 出力 first5 は `.52`（π0.5 `.46`）。行動誤差で選ぶ上限は、先頭5下書き `.44`、単位境界ごとに最良をつなぐと `.32`。これらは正解を参照する上限診断で、実行時の検索結果ではありません。Rerank offline の主処理は run log の約10分、bridge追加は数分。診断の個別時間を記録していない箇所は未記録扱いです。

### 12. atomic_seen 18タスクの準備

```bash
bash "$S/atomic18-prepare.sh"
```

保存済みの `memory-18` は18タスク×100デモから U/W メモリと文脈特徴を作成済み（memory 80 / heldout 20）。総 U 単位は task summary を確認してください。キー・Expert 学習および18×50閉ループは実行されていません。特徴抽出の所要時間は完了ログに壁時計記録がないため未記録です。

## 新規 clone での確認コマンド

本書の実装確認は重いGPU処理を含めません。clone root から次を実行します。

```bash
export SKILLNET_STATE_DIR="${SKILLNET_STATE_DIR:-$HOME/.skillnet-state}"
export SKILLNET_VENV="${SKILLNET_VENV:-$SKILLNET_STATE_DIR/envs/skillnet}" # 別途用意した SkillNet venv
export PATH="$SKILLNET_VENV/bin:$PATH"
export SKILLNET_ROBOCASA_CHECKPOINT="$SKILLNET_STATE_DIR/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000"
export PYTHONPATH="$(pwd)/skill_moe/skillnet/src:$(pwd)/skill_moe/skillnet/packages/skillnet-client/src:$(pwd)/skill_moe/skillnet/examples/robocasa:$(pwd)/skill_moe/skillnet/examples/memory_pilot/robocasa${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export SKILLNET_REPO_ROOT="$(pwd)"
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
taskset -c 8-23 nice -n 10 python -c 'import torch,sys,pytest; torch.set_num_threads(8); sys.exit(pytest.main(["-q", *sys.argv[1:]]))' tests/test_robocasa*.py
for f in skill_moe/skillnet/examples/robocasa/scripts/*.sh; do bash -n "$f"; done
for f in skill_moe/skillnet/examples/robocasa/scripts/*.py; do taskset -c 8-23 nice -n 10 python -m py_compile "$f"; done
taskset -c 8-23 nice -n 10 python - <<'PY'
import importlib.util, pathlib, torch
torch.set_num_threads(8)
root = pathlib.Path("skill_moe/skillnet/examples/robocasa/scripts")
for i, path in enumerate(sorted(root.glob("*.py"))):
    spec = importlib.util.spec_from_file_location(f"robocasa_script_check_{i}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
print("imported", len(list(root.glob("*.py"))), "python scripts")
PY
for f in expert2_prep.py expert2_train.py expert2_validate.py expert3_train.py expert3_validate.py; do
  taskset -c 8-23 nice -n 10 python "skill_moe/skillnet/examples/memory_pilot/robocasa/$f" --help >/dev/null
done
taskset -c 8-23 nice -n 10 python skill_moe/skillnet/examples/memory_pilot/robocasa/draft_ablation.py --help >/dev/null
taskset -c 8-23 nice -n 10 python skill_moe/skillnet/examples/robocasa/scripts/key-diag.py --help >/dev/null
```

新規 clone に venv は含まれないため、上では別途用意した SkillNet venv と取得済み checkpoint を指定します。LIBERO がない SkillNet venv では `env/activate.sh` が LIBERO 設定で終了するため、テストだけなら venv の Python と clone の `PYTHONPATH` を直接使います。シェル wrapper は `bash -n` で構文を検査し、実験を起動しません。Python CLI は help 表示だけを確認します。環境・データ・checkpoint の準備は [robocasa365.md](../../robocasa365.md) を参照してください。
