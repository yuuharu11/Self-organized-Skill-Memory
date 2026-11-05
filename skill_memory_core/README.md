# Skill Memory Core

π0.5 系の行動チャンクを生のまま保存し、M2 の顕著トークン検索で実演を1本選び、固定済みの Action Expert が下書きを flow bridge で修正する方式を、持ち運べる Python パッケージとしてまとめたものです。Python 標準ライブラリ・NumPy・PyTorchで、メモリ、検索、M6単位化、bridge数式、指標を利用できます。OpenPI/JAX/Flax は明示して adapter を呼ぶ場合だけ必要です。

## データの流れ

```text
エピソードの状態・行動
  ├─ stride 10 / horizon 10 → MotionMemory(row_id, raw action[10,32], task, episode)
  ├─ prefix tokens[512,2048] + context[79] → M2 pooled key + salient token/weight
  └─ query key cosine → task除外 coarse Top-64 → bidirectional weighted MaxSim → row ID Top-1
                                                        │
                               actions[dataset_row_id] ─┘  (行動を平均しない)
                                                        ↓
               x(t0)=(1−t0)draft+t0·noise → 固定 Expert の Euler bridge → action[10,32]
```

## 移植手順

1. `pip install ./skill_memory_core` で入れるか、`skill_memory_core/` を別プロジェクトへコピーして親ディレクトリを `PYTHONPATH` に追加します。`import skill_memory_core` は OpenPI を読み込みません。
2. データローダー側で各区間に一意のデータセット行ID、正規化行動 `[N,10,32]`、task ID、episode ID を割り当てて `MotionMemory` を作ります。候補IDは常にメモリ内オフセットでなくデータセット行IDです。
3. 同一バージョンの M2 重みがある場合、prefix由来の pooled key、token hidden、salience weight を保存します。token hidden/weight は保存時に fp16 に丸めます。`SalientTokenKey.load_production_weights()` は M2 state_dict 名を維持します。
4. coarse cosine検索と MaxSim再順位付けで候補行IDを得て、`drafts_by_row_id(dataset_actions, candidate_ids)` で実行下書きを取得します。タスク・エピソード除外は呼び出し側の分割に合わせ、候補を検査します。
5. OpenPI π0.5 を使う場合だけ `skill_memory_core.adapters.openpi_pi05` を import し、モデル、base checkpoint、partial Expert weights、Observation作成を呼び出し側で渡します。学習データ・除外suite・checkpoint選択は呼び出し側が決めます。

## インターフェース

| 部品 | 入出力・要点 |
|---|---|
| `MotionMemory` | `row_ids[N], actions[N,H,D], tasks[N], episodes[N]`。NPZ保存/読込、task/episode除外、dataset row IDから下書き取得。 |
| `SalientTokenKey` | tokens `[B,512,2048]`, context `[B,79]`, mask `[B,512]`。dim=96。M2 `state_dict` と互換。 |
| `Retriever` | query key/token特徴から coarse cosine Top-K、任意に weighted MaxSim rerank。返却値は dataset row ID とスコア。 |
| `build_context` | `as_trained` は既存重みに一致する歴史的配置、`fixed` は意図された `[state8,previous action70,flag]`。 |
| `sample_rank_mixture` | rank1 .70、rank2–5各 .075 で実在候補を1本サンプル。動作を平均しない。 |
| `m6_fit` / `medoid` | 生のactive7 action MSEを機能距離にし、ε・support・episode数で可変個数の単位を作る。代表は実在区間medoid。 |
| `bridge_batch` / `integrate_euler` | flow bridgeの訓練点・目標速度と `t0→0` 積分。NumPy式はJAX/Torch演算にも対応。 |
| `adapters.openpi_pi05` | 遅延 import のπ0.5モデル・prefix特徴・velocity・sample_bridge・Expert overlay/学習補助。 |

## 主要パラメータ（現行の再現値）

| 項目 | 値 |
|---|---:|
| 行動単位 | horizon=10、stride=10、action dim=32（誤差はactive7） |
| M2 token key | token dim 2048、token count 512、context 79、投影dim 96 |
| 粗検索 | cosine Top-64、同一task除外、同点はdataset row ID順 |
| MaxSim | query→memory と memory→query をsalienceで重み付け、平均0.5/0.5、同点は安定順 |
| 保存丸め | token hidden と salience weight をfp16、検索計算はfp32 |
| bridge学習 | `t0~U(.2,.6)`, progress `~U(.01,1)`, 10-step action、active7速度MSE |
| Expert optimizer | AdamW 2.5e-5, clip norm 1、action expert trainable filter |
| Expert評価 | t0候補 .2/.4/.6、noise seed 4101、10 Euler steps |

## 証拠の現状

LIBERO-40のLong（task 0–9）では、検索キーの M2 Top-1 生行動MSEは全行 `.105`、hard `.156`。同じ30-taskメモリの action oracle 上限は `.029` / `.035` で、検索の誤差が大きく残ります。別々に公平学習した Expert の評価では、E-M2＋M2 Top-1 は E-zero＋zero より良く（F1成立）、ただし E-M2＋正しいM2候補はシャッフル候補を有意に上回らず（F2不成立）。したがって「メモリで学習したExpertが有用」と「検索した動きの中身が効いた」は区別が必要です。閉ループでこのCoreを使った評価は未検証です。公開π0.5のモデルとVLMはLong taskを事前学習で見ています。

## 既知の注意

- MaxSim の後段は直接学習していません。学習損失は pooled key の multi-positive InfoNCE だけです。
- 公開M2の更新は約512 update、キーと Action Expert は別々に学習し、保存済み実験は1 seedです。
- `_history` の文脈生成に既知のslice不具合があります。`as_trained` は再現互換として残し、`fixed` に切替えても既存重みとは互換でありません。保存済み結果への影響は未確認なので再学習なしに変更しないでください。
- 閉ループ向け実時間検索サーバーは未実装です。履歴特徴を作るなら、物理状態でなく実際に実行済みの直前行動を渡してください。
- M6は単位数がデータ依存です。単位代表1本に縮約すると、全メンバー検索より最近傍距離が悪化する場合があります。
- 正規化、task/episode IDの採番、カメラtoken順、prefix mask、action列の列順を元データと揃えてください。異なる規約はcheckpoint parityを壊します。

## 元実装 → このパッケージ

| 元の場所 | 新しいAPI |
|---|---|
| `pi05_memory/core.py`: `normalize_tokens`, `late_interaction`, `bridge_batch` | `key.normalize_tokens`, `bridge.bridge_batch` |
| `pi05_memory/core.py`: `MemoryPi05.prefix_features/velocity/sample_bridge` | `adapters.openpi_pi05.make_model` が返す `MemoryPi05` |
| `pi05_memory/run.py`: `load_rows`, `obs_batch`, `load_policy`, `make_bridge_step` | データI/Oは呼出側、π0.5 load/stepは `adapters.openpi_pi05` |
| `key_mechanism/models.py`: `M2SalientTokens` | `key.SalientTokenKey` |
| `key_mechanism/run.py`: `_history`, `_positive_pairs`, `_contrastive_loss`, `_retrieve`, `_m2_rerank` | `key.build_context`, `positive_pairs`, `infonce_loss`, `coarse_search`, `maxsim_rerank` |
| `fair_expert/run.py`: M2 rank抽選、別task shuffle、候補検査 | `drafts.sample_rank_mixture`, `shuffle_cross_task`, `validate_candidates` |
| `skill_expert/run.py`: partial weights overlay、trainable filter、bridge update | `adapters.openpi_pi05.load_expert`, `trainable_filter`, `make_bridge_step`, `train_expert` |
| `skill_memories/build.py`: `RawFunctional`, `m6_fit`, `unit_medoid` | `units.RawFunctional`, `m6_fit`, `medoid` |

対象外の旧実装と理由: M1/M3は比較キー・候補judge、codec b12/M1は圧縮表現、bpriorは潜在prior検索、adaptiveは信頼度制御、rescueは失敗後branchingであり、このパッケージの「生区間＋M2＋修正Expert」主経路に含めません。出典はそれぞれ `pi05_memory/key_mechanism/models.py`、`pi05_memory/` のcodec/bprior/adaptive/rescue各サブディレクトリです。
