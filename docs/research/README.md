# 研究ノート

環境構築・データ処理の手順とは別に、進行中の研究の方針・判定基準・調査をまとめたもの。

初めて読む場合は、[再現ガイド](../reproduction.md) →
[スキルメモリの設計](anchor_flow_skill_memory.md) → [測定結果](skill_memory_findings.md) の順を勧めます。
再現ガイドには公開コードの対象範囲と必要な入力資産を、
[スクリプト一覧](../../skill_moe/skillnet/examples/emergent/light/intermediate/README.md) には実験ごとの入口を記載しています。

ノート中の日時付き run 名や `.skillnet-state/` へのリンクはローカルの実行記録です。
Git に含まれないファイルや未コミットの新しい実験を、このチェックアウトだけで再現できるとは扱いません。

## 創発スキル（現在の主線）

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

- [skill_representation_review.md](skill_representation_review.md) — 追加可能なスキルメモリの表現学習
- [prior_work_and_baselines.md](prior_work_and_baselines.md) — 先行研究と直接の比較対象（入口。主張ごとの対応と、実行する baseline の一覧）
- [related_work_retrieval_memory.md](related_work_retrieval_memory.md) — 関連研究とRSS投稿判断
- [novelty_anchor_flow.md](novelty_anchor_flow.md) — 開始点デノイズの新規性と実現性

## 比較対象と実装

- [baselines_success_rates.md](baselines_success_rates.md) — 公開値の比較条件とタスク集合
- [optimus_r_reproduction.md](optimus_r_reproduction.md) — Optimus-R の仕様調査と独自実装の範囲
- [S2 スクリプト一覧](../../skill_moe/skillnet/examples/memory_pilot/anchor_flow/README.md) — 較正・検索・区間スキル・場面適応の入口

## レビュー

- [reviews/review_representation.md](reviews/review_representation.md) — スキル表現と「正しいスキル」の定義
- [reviews/review_vla_closed_loop.md](reviews/review_vla_closed_loop.md) — S2 閉ループ（学習目標の不具合の発見）
- [reviews/review_external_s2_pipeline.md](reviews/review_external_s2_pipeline.md) — 外部レビュー：S2 パイプライン（キーの平均化、温度、中心への丸め）と、ログでの検証
- [reviews/morning_report.md](reviews/morning_report.md) — 夜間の自律実験の朝のレポート（2026-10-30）
- [reviews/experiment_results_2026-10-30.md](reviews/experiment_results_2026-10-30.md) — 実験結果のまとめ（手法の定義と全結果の数値、単独で読める）
- [reviews/handoff_2026-10-30.md](reviews/handoff_2026-10-30.md) — 測定停止時点の引き継ぎ（英語、過去の実行条件）

開発環境は [../development_environment.md](../development_environment.md) を参照。
