# スキル潜在 z の評価入力形式（schema v1）

`examples/emergent/eval/` の評価・メモリ部品が読む形式。モデルには依存しない。
既存の記録形式（[docs/research/memory_pilot_gates.md](../../../../../docs/research/memory_pilot_gates.md)「保存形式」、
[g2_retrieval/SCHEMA.md](../../memory_pilot/g2_retrieval/SCHEMA.md)）の**拡張**であり、既存キーの意味は変えない。
したがって z の記録は、そのまま G2 の `load_run` で読める記録でもある。

## ディレクトリ

```
<run_dir>/manifest.json
<run_dir>/episodes/<episode_id>.npz      # 推論呼び出しごとに1行（N行）
```

## manifest.json

既存のキー（`format_version`, `suite`, `replan_steps`, `env_step_offset`, `episodes[]` の
`episode_id`/`task`/`task_id`/`suite`/`init_state_id`/`success`/`steps`）はそのまま。
追加で `emergent` ブロックを書く。

| キー | 必須 | 内容 |
| --- | --- | --- |
| `emergent.schema_version` | 必須 | `1` |
| `emergent.z_horizon` | 必須 | z が表す区間の長さ H（環境ステップ数。本命 50、比較 25） |
| `emergent.z_dim` | 必須 | z の次元 |
| `emergent.model` / `checkpoint` | 必須 | モデル名とチェックポイントのパス・リビジョン |
| `emergent.untrained_checkpoint` | 任意 | 未学習基準（`z_untrained_*`）を作ったモデルの初期化シード等 |
| `emergent.posterior_input` | 必須 | Enc が見たもの（例: `"executed actions [step, step+H) + proprio at step"`） |
| `emergent.prior_input` | 必須 | Prior が見たもの（例: `"images + proprio at step, no language"`） |
| `emergent.memory_z` | 任意 | 記録側でメモリの検索・ブレンド z も出す場合の設定（評価側は自前で store 分割から作る） |
| `emergent.key` | 必須 | キー特徴の由来。`{"vlm_img_nolang": {"layer":..., "tokens": "image", "pooling": "mean", "language_input": false}, "vlm_img_lang": {..., "language_input": true}}` |
| `emergent.training_episodes` | 任意 | モデルの学習に使ったエピソード／デモの一覧（評価行と重ならないことの確認用） |

## episodes/<episode_id>.npz に追加するキー

N は推論の数（既存の `step`, `proprio`, `pred_chunk`, `n_executed` と同じ行）。

| キー | 形 | 必須 | 内容 |
| --- | --- | --- | --- |
| `z_post_mean` | N×Dz float | 必須 | Enc(行動チャンク, 固有感覚) の平均 |
| `z_post_logvar` | N×Dz float | 必須 | Enc の対数分散 |
| `z_prior_mean` | N×Dz float | 必須 | Prior(観測) の平均 |
| `z_prior_logvar` | N×Dz float | 任意 | Prior の対数分散（あれば KL(post‖prior) も出す） |
| `z_untrained_post_mean` / `z_untrained_prior_mean` | N×Dz float | 任意 | 同じ入力を未学習（初期化のみ）のモデルに通した z。MI の基準 |
| `z_interval_len` | N int | 必須 | 事後 z の区間 `[step, step+H)` のうち実際に実行された環境ステップ数（エピソード終端で H 未満） |
| `z_executed` | N bool | 必須 | その推論のチャンクが実行されたか（`n_executed>0`） |
| `key__vlm_img_nolang` | N×Dv float | 必須 | **言語を入力せずに** VLM を通したときの画像出力トークンのプール（本命のキー） |
| `key__vlm_img_lang` | N×Dv float | 必須 | 通常の入力（言語あり）で VLM を通したときの画像出力トークンのプール（比較用のキー） |

π0.5 の prefix では画像トークンと言語トークンが双方向に注意し合うため、通常入力の画像トークンにも
言語の情報が混ざる。そのため「画像トークンだけをプールする」では言語なしにならず、言語を入力しない
VLM の実行を別に行う。キー = 2種のどちらか＋固有感覚（`proprio`）で、`KeyEncoder.fit(include_language=...)` で切り替える。
| `z_target_actions` | N×H×Da float | 任意 | Enc が見た「きれいな行動チャンク」= 実行された行動 `[step, step+H)`。無ければ `pred_chunk[:n_executed]` を区間に渡って連結して作る |
| `z_target_mask` | N×H bool | 任意 | `z_target_actions` の有効ステップ |

規則:

- `z_post_*` は**実行された**行動（デモまたは実際に実行された軌道）からだけ作る。実行されなかった予測の
  末尾を Enc に入れない。`z_interval_len < H` または `z_executed=False` の行の事後 z は NaN でよい
  （その行はメモリに登録されず、評価でも使われない）。それ以外の z とキーは有限値であること。
- `z_prior_mean` とキーは推論時点の観測だけから作る（その行の行動・未来を見ない）。
- 固有感覚は既存の `proprio` を使う（キー用に別に保存しない）。
- 評価で使う運動ラベル（環境の手先軌道 `env_*`）、タスク文字列、`skills`/`skill_mask` は**評価専用**。
  学習・Enc・Prior の入力に使わない。`skills`/`skill_mask` は新モデルが使わない場合でも、
  タスク注釈の計画をそのまま記録しておく（サブタスクをまたいだ検索の評価に要る）。
- 行の時刻は既存と同じ: `step` は観測時の環境ステップ、環境配列の添字は `step - env_step_offset`。

## 評価側の扱い（`data.load_z_run`）

- 評価に使う行 = `z_executed` かつ `z_interval_len == H` かつ 環境配列に H ステップ先まである行。
- 運動の特徴（`metrics.motion_labels`）:
  - **env**: 実際に起きた手先の軌道 `env_*` の、区間終端での変位（支配的な符号付き軸 7 クラス）、
    回転（同 7 クラス）、グリッパー（2本指の開き幅 `q_left - q_right` の変化が ±0.01 を超えるか、3 クラス）。
    primitives と同じ規則。
  - **action**（自明性の参照）: Enc の入力そのもの（`z_target_actions`）から作った同じ 3 種。
    事後 z とこのラベルの MI は、ボトルネックが通した入力情報の量であり、創発の証拠ではない。
- go/no-go は**推論時に実際に使う z**（Prior の z と、メモリから検索・ブレンドした z）で判定する。
  Enc の事後 z は参考値。判定規則（(i) Prior z の MI、(ii) 他タスク検索で gap の50%以上、(iii) 検索・ブレンド z の再構成誤差 < Prior z）は `gonogo.py` の docstring。実行は `python -m skill_moe.skillnet.examples.emergent.eval.pipeline <run_dir>`。
- 既存記録で試す場合は `aliases={"z_post_mean": "feat__L09_mean_t_last", ...}` で読み替える
  （`z_post_logvar` が無い場合は 0、`z_interval_len`/`z_executed` は `n_executed` と環境配列から作る）。
