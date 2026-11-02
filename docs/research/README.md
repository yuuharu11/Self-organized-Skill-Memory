# 研究ノート

環境構築・データ処理の手順とは別に、進行中の研究の方針・判定基準・調査をまとめたもの。

現在の研究判断は[停止記録とSkillNetの失敗診断](reviews/diagnosis_2026-11-02.md)から読む。
別環境で動かす場合は[再現ガイド](../reproduction.md)を参照する。
再現ガイドには公開コードの対象範囲と必要な入力資産を、
[スクリプト一覧](../../skill_moe/skillnet/examples/emergent/light/intermediate/README.md) には実験ごとの入口を記載しています。

ノート中の日時付き run 名や `.skillnet-state/` へのリンクはローカルの実行記録です。
Git に含まれないファイルや未コミットの新しい実験を、このチェックアウトだけで再現できるとは扱いません。

## 現在の方針（2026-11-02）

- [停止記録とSkillNetの失敗診断](reviews/diagnosis_2026-11-02.md) — **最新**。A・DMPを保留し、言語・段階移行・操作の原因を分けて調べる。360行の途中結果を固定して再集計。
- [夜間のオフライン結果](reviews/overnight_2026-11-02.md) — 検索・区間スキル・5-shot対照。入力特徴差と評価条件を訂正済み。
- [DMP設計の履歴](dmp_routing_plan_2026-11-02.md) — 69/135試行でユーザー指示により停止。実施指示ではない。
- [診断の再集計スクリプト](../../skill_moe/skillnet/examples/skill_conditioning/summarize_rollout_diagnostics.py) — 固定JSONLの共通組だけを比較。GPU・重みは不要。

## 創発スキルの設計履歴（実験再開は保留）

1. [emergent_skill_plan.md](emergent_skill_plan.md) — 方針
2. [emergent_skill_verification.md](emergent_skill_verification.md) — 最初の検証のアルゴリズムと判定基準
3. [skill_formation_study.md](skill_formation_study.md) — スキル表現の作り方の比較（軽量版）
4. [skill_construction_brief.md](skill_construction_brief.md) — スキルの作り方の暫定方針
5. [skill_memory_findings.md](skill_memory_findings.md) — 軽量検証のこれまでの結果
6. [anchor_flow_skill_memory.md](anchor_flow_skill_memory.md) — 運動の型から始めるデノイズ：主張と検証計画（S1 軽量版、S2 LIBERO 閉ループ）
7. [architecture_v2.md](architecture_v2.md) — 結果を踏まえた改善案と、各部品の設計の根拠

## 別軸の候補（インターン向け）

- [skill_conditioning_fixes.md](skill_conditioning_fixes.md) — 既存スキル系VLAの構造的な弱点と小さな改善。外部エージェントの案の検証、順位、シミュと実機のタスク設定
- [外部から参照する手順](../../skill_moe/skillnet/examples/skill_conditioning/README.md) — このリポジトリ外（upstream の SkillNet）で事実を確かめる手順、計測スクリプト、最初の実験 T0

## 検索メモリ（前段の事前判定）

- [memory_pilot.md](memory_pilot.md) — 初期実験の方針
- [memory_pilot_gates.md](memory_pilot_gates.md) — 判定実験 G0〜G3
- [libero_lifelong_protocol.md](libero_lifelong_protocol.md) — LIBERO生涯学習の評価プロトコル

## 調査

- [related_action_units_2026-11-02.md](related_action_units_2026-11-02.md) — 可変長行動単位、潜在スキル、既存方式との重なり
- [related_dmp_routing_2026-11-02.md](related_dmp_routing_2026-11-02.md) — DMPルーティングの設計時の調査
- [skill_representation_review.md](skill_representation_review.md) — 追加可能なスキルメモリの表現学習
- [prior_work_and_baselines.md](prior_work_and_baselines.md) — 先行研究と直接の比較対象（入口。主張ごとの対応と、実行する baseline の一覧）
- [related_work_retrieval_memory.md](related_work_retrieval_memory.md) — 関連研究とRSS投稿判断
- [novelty_anchor_flow.md](novelty_anchor_flow.md) — 開始点デノイズの新規性と実現性

## 比較対象と実装

- [baselines_success_rates.md](baselines_success_rates.md) — 公開値の比較条件とタスク集合
- [optimus_r_reproduction.md](optimus_r_reproduction.md) — Optimus-R の仕様調査と独自実装の範囲
- [S2 スクリプト一覧](../../skill_moe/skillnet/examples/memory_pilot/anchor_flow/README.md) — 較正・検索・区間スキル・場面適応の入口

## レビュー・報告・ログ（`reviews/`）

最新は[11-02の停止記録と診断](reviews/diagnosis_2026-11-02.md)。過去の提案は当時の判断の記録として保存し、現在の実施方針と混同しない。

### 報告（単独で読める）

- [reviews/progress_2026-11-01.md](reviews/progress_2026-11-01.md) — 閉ループ・失敗分析・検索方式の不通過
- [reviews/answer_2026-10-31.md](reviews/answer_2026-10-31.md) — 10-31時点のレビューへの回答と速報
- [reviews/report_2026-10-31_full.md](reviews/report_2026-10-31_full.md) — 前提知識なしで読める10-31時点の報告（2026-10-31 15:20 UTC）
- [reviews/report_2026-10-31.md](reviews/report_2026-10-31.md) — 同日の先行版。§7 外部レビューへの対応と §8 1,500 行での訂正を含む（上の版に統合済み）
- [reviews/experiment_results_2026-10-30.md](reviews/experiment_results_2026-10-30.md) — 実験結果のまとめ（手法の定義と全結果の数値、場面に合わせる変換器 F）
- [reviews/morning_report.md](reviews/morning_report.md) — 夜間の自律実験の朝のレポート（2026-10-30）

### 査読・相談

- [reviews/proposal_R012_2026-11-01.md](reviews/proposal_R012_2026-11-01.md) — R0/R1/R2提案の履歴
- [reviews/proposal_final_2026-11-01.md](reviews/proposal_final_2026-11-01.md) — 69タスクの検索・生成方式の決着実験の提案と結果
- [reviews/direction_B_2026-11-01.md](reviews/direction_B_2026-11-01.md) — 未来運動で型を作る方向の検討履歴（現在は保留）
- [reviews/review_representation.md](reviews/review_representation.md) — スキル表現と「正しいスキル」の定義
- [reviews/review_vla_closed_loop.md](reviews/review_vla_closed_loop.md) — S2 閉ループ（学習目標の不具合の発見）
- [reviews/review_external_s2_pipeline.md](reviews/review_external_s2_pipeline.md) — 外部レビュー：S2 パイプライン（キーの平均化、温度、中心への丸め）と、ログでの検証
- [reviews/consultation_2026-10-30.md](reviews/consultation_2026-10-30.md) — 外部モデルへの相談：新しいタスクでのスキルの取り出し

### 作業ログ・引き継ぎ

- [reviews/night_log_2026-10-30.md](reviews/night_log_2026-10-30.md) — 10/30 夜〜10/31 の自律作業の経緯と判断、全表
- [reviews/night_log_2026-10-31.md](reviews/night_log_2026-10-31.md) — 10/31午後〜11/01朝の作業ログ（履歴）
- [reviews/handoff_2026-10-30.md](reviews/handoff_2026-10-30.md) — 測定停止時点の引き継ぎ（英語、過去の実行条件）

開発環境は [../development_environment.md](../development_environment.md) を参照。
