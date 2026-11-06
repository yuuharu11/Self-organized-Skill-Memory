# 07 π0.5 LIBERO-40：運動メモリと修正 Expert（進行中）

## 問い

公式 `pi05_libero` を固定し、トークン単位の MaxSim 検索で取り出した運動（Top-1）を下書きにして、独立に学習した修正用 Action Expert でデノイズすると、同じ π0.5 の通常生成より成功率が上がるか。公開値（SkillNet LIBERO-40 の Long 98.4%）は参考値で、同一条件の再現ではない。

## 状態

- 学習・校正・閉ループ評価までのパイプラインを接続済み。評価はまず 4 suite × 10 タスク × 5 初期状態 × 通常 / 提案。
- 圧縮器（M1、b=12、r=4）の診断：b は再構成に使われているが、r の寄与はほぼない。b と r に分離できた根拠にはしない。
- 検索キー監査は完了。全3,277 queryを共通Oracleで評価し、VLM由来・非VLM由来キー、π0.5予測行動、codec prior、予測誤差の大きいEval部分集合を比較した（詳細は下記）。
- 対象タスクを学習済みのモデルなので、未知タスクへの転移実験ではない。

## まず読むもの

- [diagnosis.md](diagnosis.md)：メモリが成功率を上げない原因の診断（保存・検索・ノイズ・修正Expertの切り分けと結論）

## ファイル

| ファイル | 中身 |
|---|---|
| [design_and_status.md](design_and_status.md) | 方式、重みとデータ、後から変えられるもの、実行・監視・圧縮器の診断 |
| [retrieval_correspondence/results.md](retrieval_correspondence/results.md) | 3,277 query のVLM・非VLM検索キー統合評価、検索なし基準、高予測誤差部分集合 |
| [summary.md](summary.md) | 評価条件・オフライン検索／下書き診断・閉ループ結果の総括 |
| [paired_breakdown/results.md](paired_breakdown/results.md) | 通常方策との対応付き救済・悪化、通常失敗試行一覧 |
| [topk_long/plan.md](topk_long/plan.md) | LIBERO-10 Long限定・生行動Top-K候補の生成後診断計画と事前判定 |

スクリプトは [`examples/memory_pilot/pi05_memory/`](../../../skill_moe/skillnet/examples/memory_pilot/pi05_memory/)。データから評価までの順序は [再現手順](../../reproduce_core_path.md) を参照。実行結果は `.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/`（Git 管理外）。
- [修正 Expert の下書き依存性](draft_sensitivity/results.md)：同じノイズで下書きだけを変え、追加学習と下書き内容の効果をオフライン行動誤差で切り分ける。
- [圧縮行動と生行動の公平比較](compress_vs_raw/README.md)：同じM2検索候補を使い、raw／M1-b12下書きで学習した修正ExpertをLong Evalで比較。
- [メモリ有効候補の診断](memory_gap/results.md)：難しい Eval 行について検索・圧縮・下書き利用をオフライン比較。
- [生行動に近い候補の検索](retrieval_raw/README.md)：VLM、手首＋状態、短い履歴、学習キーで段階4-Bを評価。
- [失敗状態からの分岐検証](rescue_branching/results.md)：4キー×圧縮/生を含む44条件の本番を、通常失敗10試行・30分岐点で実行中。
- [行動メモリ表現の診断](representation/results.md)：同じ行動oracle候補を固定し、raw・M1 bottleneck・EEF相対・状態差分をオフライン比較。
- [b予測下書きのLong閉ループ](bprior_long/results.md)：写像b̂を直接下書きにする／b̂で検索する2条件、各100試行。
- [信頼度適応のLong閉ループ](adaptive_long/results.md)：検索スコアで開始ノイズを変える条件と、通常π0.5への切り替え付き、各100試行。
- [複数の運動メモリ](skill_memories/README.md)：raw行メモリとM6の可変数medoid単位をLong除外で構築し、Long Evalへの候補距離をCPU診断。
- [可変長区間の追従診断](varlen/README.md)：区間開始での一度の検索と固定10-step毎の再検索をLong Evalで比較。
- [Skill ExpertのLong閉ループ](skill_expert/results.md)：スキルbを復元した下書きから直すExpertと、写像b̂の信頼度でt0を変える条件、各100試行。
- [局所キーの検索](local_key/README.md)：手先まわりのキーがLong以外の実演から未知タスクの動きを検索できるかのオフライン評価。
- [キー選択機構](key_mechanism/README.md)：アテンション型のキー・候補判定器で単一実演を選び、Longの生成前後MSEとK1–K4を評価。
- [圧縮とクラスタリングの上限診断](compress_cluster_oracle/results.md)：正解を必ず含むメモリから正解で選び、検索の影響を除いて、圧縮とクラスタリングの損失とノイズ強度ごとのデノイズ後MSEを比較。
- [移植用アルゴリズムCore](../../../skill_memory_core/README.md)：生行動メモリ＋M2検索＋bridge Expertの自己完結パッケージと保存済み実験へのparity表。
