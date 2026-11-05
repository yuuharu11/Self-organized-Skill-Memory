# SkillNet-90 → LIBERO-Long 互換性確認

確認日: 2026-11-04。目的は既存成果物・コード・shape metadata の読み取り確認のみ。重みの復元、学習、特徴抽出、rollout、大規模変換は実施していない。既存の `plan.md` は変更していない。

## 1. 「47%」の出所

出所は `.skillnet-state/outputs/memory-pilot/abstract-prompt-20261101/` の公開 SkillNet 評価から切り出した途中スナップショットである。固定記録は [`skillnet_prompt.jsonl`](../05_skillnet_diagnosis/evidence/skillnet_prompt.jsonl) と [`skillnet_prompt_summary.json`](../05_skillnet_diagnosis/evidence/skillnet_prompt_summary.json)、解釈は [`dmp_stop_and_diagnosis.md`](../05_skillnet_diagnosis/dmp_stop_and_diagnosis.md) §2。

| 項目 | 確認結果 |
|---|---|
| 条件・成功数 | L0（元のタスク指示文）28/60 = 46.7%（通称「47%」）。60組すべて `complete`。 |
| タスク・初期状態 | `libero_skill_obj` の9タスク。共通ペアは task 1–6 で init 0–6、task 7–9 で init 0–5（計60）。10タスク×10初期状態のLIBERO-Long全件評価ではない。 |
| 実行設定 | 途中スナップショットは max 800 env control steps、初期安定化10 step、replan 5 step、action horizon 10、環境seed 7、trial seed は 7+init。query seed は `(7+init)*100000+query_sequence`。設定は `manifest.json`、実測の60行は投影JSONLで確認。 |
| スキル入力 | L0 は prompt は原文だが `skill_mode=oracle`。例 `task02/init-00/L0/attempt-1/result.json` には `skills=[3,0,1]`、`objects=[0,1,1]` が記録される。したがってこの47%は、正解のスキル・物体注釈を入力する条件であり、注釈なしの実行成績ではない。 |
| 後続結果 | 実験完了後の900行では L0=46/90=51.1%（同じ8百step cap）。47% は途中60組の固定値で、最終値ではない。 |

名前に「Long」と付けて引用するのは不正確である。ここで追跡できた47%の母集団は9タスクの Skill-Obj 評価であり、LIBERO-Longの10タスク試行ではなかった。したがって本移行の成功率ベースラインには流用できない。再評価は行っていない。

## 2. SkillNet-90 checkpoint の abstract tree

指定どおり次の config から `nnx.eval_shape` で abstract model を構築し、Orbax の checkpoint metadata と leaf path・shape を照合した。params の実配列は restore していない。

```python
Pi0Config(
    pi05=True, action_horizon=10, discrete_state_input=False,
    paligemma_variant="gemma_2b", action_expert_variant="gemma_300m_moe_4",
)
```

結果: abstract tree 69 leaves、checkpoint metadata 69 leaves、パス差 0、shape mismatch 0。したがって **モデルの構造・shape は一致**し、dense `openpi.models.pi0.Pi0` からは一致しない SkillNet-90 の木を、指定 MoE SkillNet config で受けられる。

config のその他の実値は `action_dim=32`、`skill_num=6`、`skill_embed_dim=64`、`positional_encoding=False`、`t_coding=False`、`dtype=bfloat16`、`vision_variant=So400m/14`。shape metadata 照合は重み値、チェックポイントの実推論、入力契約まで保証するものではない。

確認コマンド（`skill_moe/skillnet` から実行、CPU のみ）:

```bash
source env/activate.sh
taskset -c 8-47 nice -n 10 env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python - <<'PY'
import jax, flax.nnx as nnx, orbax.checkpoint as ocp
from pathlib import Path
from flax.traverse_util import flatten_dict
from openpi.models.pi0_config_moe_skill import Pi0Config
cfg = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False,
                paligemma_variant='gemma_2b', action_expert_variant='gemma_300m_moe_4')
model = nnx.eval_shape(lambda: cfg.create(jax.random.key(0)))
abstract = flatten_dict(nnx.state(model, nnx.Param).to_pure_dict())
metadata = ocp.PyTreeCheckpointer().metadata(Path('.skillnet-state/checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999/params'))
leaves = flatten_dict(metadata.tree['params'])
checkpoint = {k[:-1]: v.shape for k, v in leaves.items() if k[-1] == 'value'}
assert set(abstract) == set(checkpoint)
assert all(tuple(abstract[k].shape) == tuple(checkpoint[k]) for k in abstract)
print(len(abstract), len(checkpoint), 'leaves; no path or shape mismatches')
PY
```

実行結果: `69 69 leaves; no path or shape mismatches`。JAX CPU 初期化時に CUDA plugin 登録の警告は出たが、`JAX_PLATFORMS=cpu` で形状計算はCPU実行、GPU利用・重み読み込みなし。

## 3. 下書き開始からのデノイズを接続する変更

現行の [`pi05_memory/core.py`](../../../skill_moe/skillnet/examples/memory_pilot/pi05_memory/core.py) `make_model()` は dense `Pi0` を継承する `MemoryPi05` を作る。`sample_bridge()` は `x=(1-t0)*draft+t0*epsilon` から `t0→0` を積分するが、`velocity()` の prefix/suffix forward に SkillNet の skill embedding や `skill_mask` を渡していない。従って dense 実装のクラス継承を差し替えるだけでは、checkpointを復元できず、復元後もSkillNetと同じMoE条件の推論にならない。

関数単位の変更・接続点は次のとおり。

1. **モデル生成・復元** — `core.py::make_model` を `pi0_config_moe_skill.Pi0Config` と `pi0_moe_skill.Pi0` で構築する。checkpoint は今回一致確認した config で strict restore し、leaf の欠落・追加を初期値で埋めない。
2. **Observation 作成** — `core.py::obs_batch` と policy transform/server の入力構築で、実行時の `skills` と `skill_mask` を `[B,6]` 契約で埋める。現行 `obs_batch` は画像・state・prompt までしか作らない。`Observation` にフィールドがあること自体は、実行時の値の生成方法にはならない。
3. **prefix/key** — `prefix_features` は検索キーを作る。SkillNetのMoE prefix forwardでは `skill_embeds = llm(skills, method='skill_embed')` と `skill_mask` を渡す。検索キーは検索対象の候補を選ぶ前に使える画像・指示・状態だけで定義する必要がある。skill 条件つき prefix cache と、skill 無条件の検索特徴を同じcacheとして使わない。
4. **bridge velocity** — `embed_suffix(obs, noisy_actions, timestep)` は行動tokenと Pi05 の adaRMS 時刻条件を作る。各積分stepの MoE `llm` 呼び出しにも `skill_embeds`、`skill_mask`、`adarms_cond=[None, cond]` を渡し、`action_out_proj` で32次元速度を出す。
5. **sampler API** — `sample_actions` に `draft/x_src` と `t0` を受ける bridge branch（または別 `sample_actions_from_anchor`）を設ける。t0=1 の既存noise samplerとの数値互換を残し、t0<1 は `x_src=(1-t0)*draft+t0*epsilon`、`N` stepで `x -= (t0/N)*v(x,t)` を実装する。draft・noise・stateは同一のaction/state正規化空間であることを検証する。
6. **policy出力** — inverse normalization 後に LIBERO が受ける7次元を返す。padされた残り25次元を実行環境へ流さない。

再利用可能な先行実装として [`pi0_anchor_flow.py`](../../../skill_moe/skillnet/src/openpi/models/pi0_anchor_flow.py) は `pi0_moe_skill.Pi0` を継承し、`_integrate_from_anchor`、`sample_actions_from_anchor`、`sample_actions_from_prefix_cache` でskill embedding/maskをprefixとdenoise stepへ渡している。これは実行経路の雛形で、今回の評価や修正Expertの品質確認ではない。`core.py` dense 実装を移植する際は、この経路を再利用・統合するのが最小変更である。

**正解のskill注釈を使わない実行は、現状未定義。** `Pi0.sample_actions` は `observation.skills` をembeddingして推論し、現行 LIBERO skill policy の入力変換もデータの `skills` をID化して mask を作る。47% の L0も実際はoracle `skills/objects` を入力した。`pi0_moe_skill.Pi0.generate_subtask()` は言語token列を生成するが、その文から6種類の学習skill ID列・maskに写像する既存のrouterは確認できない。従って注釈を使わずに済ませるには、instruction/observationからID列を作る固定規則または学習済みrouterを別途用意し、その選択規則を検証する必要がある。全要素mask-offで既存モデルを呼ぶ経路はMoEのゼロskill寄与になり得るが、学習時の入力分布外なので、oracle-free代替として同等とは扱えない。

修正Expertの**構造上の追加学習は必須ではない**。既存のMoE action expertはflow timeとaction stateを受けるので、固定重みのままanchorから途中時刻で積分する小さな診断を実装できる。標準学習分布は正解actionをnoiseへ混ぜたflow pathであり、誤った検索draftからの訂正を明示的に学習したわけではない。よって「計算できる」ことと「誤ったdraftを直せる」ことは別であり、後者の有効性は要評価。既存 expert で不足が出た場合のみ、90内で作る修正Expertの追加学習が必要となる。

## 4. 行動表現・正規化・時間軸

| 項目 | 対応 | 根拠・注意 |
|---|---|---|
| モデルaction shape | 32 padded、horizon 10。LIBERO実行は先頭7次元（xyz、姿勢3、gripper）を使う。 | config `action_dim=32`, `action_horizon=10`; `LiberoOutputs` は `actions[:, :7]`。 |
| memory 40 row shape | `actions [N,10,32]`、`state [N,32]`、frameを10刻みで窓化。先頭7 action がLIBEROの実値。 | [`pi05_memory/run.py::prepare`](../../../skill_moe/skillnet/examples/memory_pilot/pi05_memory/run.py) は `PadStatesAndActions(32)`、`horizon=10`, `stride=10`。`LiberoInputs` はLIBERO行動が元からdeltaなので別delta transformをしないと説明。 |
| 学習データ時間軸 | LeRobot LIBERO-40 metadataは `fps=10`。10-frame窓は約1.0秒。 | `.../libero_40_unsegmented/meta/info.json` および `run-01/data_manifest.json` (`control_fps:10`)。 |
| 閉ループ時間軸 | `OffScreenRenderEnv` の既定 `control_freq=20 Hz`。10 step action chunkは約0.5秒、replan=5なら約0.25秒ごとに再問い合わせ。 | installed LIBERO `ControlEnv.__init__` signature。evaluationは`OffScreenRenderEnv(..., horizon=...)`を呼び、`control_freq`を上書きしない。したがって形状は同じでも、demoの10-step時間幅とrolloutの10-step時間幅は一致しない。 |
| 正規化 | 40-memoryのaction/stateはSkillNet-90 assetでなく、`run-01` が指定した公式dense `pi05_libero` checkpoint のstatsで正規化。90 modelには90 checkpointのnorm statsを使う必要がある。 | `run-01/experiment.json`, `data_manifest.json`; `run.py::prepare` は `args.checkpoint/assets/**/norm_stats.json` を一つ読み `Normalize(... use_quantiles=True)`。 |

statsファイルのSHA256は90=`fbc2950763c2b1dde59c910eb74cbd5ff47e56cd8a483720b317d232343ce447`、40 SkillNet=`db3d967b25e51b729e3e4e2d1c8c0ab2aeb9bcf9cc6f478a36027c23391693f1`、run-01 dense base=`b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`。action statsは異なる（例: 90のactive action mean先頭3は `[0.0394,0.0377,-0.0978]`、dense baseは `[0.0268,0.0889,-0.0998]`; gripper mean はそれぞれ `-0.0427` と `-0.1305`）。値を混ぜたままdecode/bridgeしてはいけない。既存40-memoryを入力にするなら、保持したnormalization metadataでraw値へ戻し90 statsへ変換するか、memory/codecを90空間で作り直す。

**結論:** active7とchunk shapeは合わせられるが、normalizationは不一致、demo/rolloutの制御周期も2倍差がある。90専用statsで90学習データを処理し、Long supportも同じ生指令空間から90規約へ変換する。続きの予測を時間的に比べる際は窓の時間長を揃える必要がある。

## 5. Longを学習に含めない場合に90側だけで作るもの

下記はLIBERO-90 train splitだけで学習・選択し、Longのquery/testや追加supportを見て更新しない。公開SkillNet-90本体checkpoint自体は固定し、public base pretrainingでLongが絶対未見とは主張しない。

1. **正規化・入力変換** — SkillNet-90の `norm_stats.json` と同じaction/state規約を固定。異なる40-memory statsをそのまま使わない。
2. **圧縮器** — action encoder/base-residual decoder/prior (例 M1 codec) を90のtrain episodes/chunksでfit。Long行をencodeするだけの追加でも、codecの学習重みとlatent尺度は90だけで決める。
3. **検索キー** — `wrist_state_contrastive` のfeature normalization、projection/contrastive weights、negative/positive selection、thresholdを90 train/devだけで作成・固定。Longからラベル・検索性能を見て学習・校正しない。Longは固定key関数でmemory key化する。
4. **修正Expert** — anchor draftを補正する学習が必要と判断する場合、90だけのtrain/heldout episodesで学習・early stopし固定。Long demoをexpert updateに使わない。追加学習をしない枝では、SkillNet-90の現expertを固定して診断する。
5. **検索bank /校正** — memory candidatesとkey bankのbase subset、confidence・noise・fallback閾値を90だけから決め固定する。追加Long memoryは保存行だけ増やす。
6. **runtime skill selector** — 正解skill annotationを禁止するなら、選択器も90内のみで訓練・固定する。現行技能ラベルの出どころが未解決のため、これは移行のblocking prerequisite。

## 見積もり・判断

指定checkpointとのshape互換と既存anchor samplerの再利用が確認できたので、モデル本体を大規模に作り直す必要はない。一方、annotation-free skill selector、90専用圧縮器/検索キー/（必要なら）修正Expert、normalization変換、時間軸をそろえた評価が残る。これらを含む最小の適切な予備実験は概算 **3–5人日**（GPU/データ処理の実時間は別）。

**安い予備実験として成立するか: 条件付き。** 固定SkillNet-90 expertで試す短い技術pilotは可能だが、正解skill注釈を使わないselectorと90規約のmemory/codecが揃わない状態では、問いに答える評価にならない。

## 参照した実装・成果物

- [`pi0_config_moe_skill.py`](../../../skill_moe/skillnet/src/openpi/models/pi0_config_moe_skill.py)、[`pi0_moe_skill.py`](../../../skill_moe/skillnet/src/openpi/models/pi0_moe_skill.py)、[`gemma_moe_skill.py`](../../../skill_moe/skillnet/src/openpi/models/gemma_moe_skill.py)
- [`pi0_anchor_flow.py`](../../../skill_moe/skillnet/src/openpi/models/pi0_anchor_flow.py)
- [`pi05_memory/core.py`](../../../skill_moe/skillnet/examples/memory_pilot/pi05_memory/core.py)、[`run.py`](../../../skill_moe/skillnet/examples/memory_pilot/pi05_memory/run.py)、[`libero_policy.py`](../../../skill_moe/skillnet/src/openpi/policies/libero_policy.py)
- `.skillnet-state/outputs/memory-pilot/abstract-prompt-20261101/manifest.json`, `cap-change-20261101T163413Z/manifest.json`, `report.md`, `analysis.json`, `trials/task02/init-00/L0/attempt-1/result.json`
- `.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01/experiment.json`, `data_manifest.json`

運用メモ: 依頼文が指定する `docs/memory_pilot_gates.md` はこの作業ツリーには存在しなかった。明示された今回の禁止事項・出力範囲・CPU-only条件に従って実施した。
