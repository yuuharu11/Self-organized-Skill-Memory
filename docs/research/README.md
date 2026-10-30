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

## 検索メモリ（前段の事前判定）

- [memory_pilot.md](memory_pilot.md) — 初期実験の方針
- [memory_pilot_gates.md](memory_pilot_gates.md) — 判定実験 G0〜G3
- [libero_lifelong_protocol.md](libero_lifelong_protocol.md) — LIBERO生涯学習の評価プロトコル

## 調査

- [skill_representation_review.md](skill_representation_review.md) — 追加可能なスキルメモリの表現学習
- [prior_work_and_baselines.md](prior_work_and_baselines.md) — 先行研究と直接の比較対象（入口。主張ごとの対応と、実行する baseline の一覧）
- [related_work_retrieval_memory.md](related_work_retrieval_memory.md) — 関連研究とRSS投稿判断
- [novelty_anchor_flow.md](novelty_anchor_flow.md) — 開始点デノイズの新規性と実現性

開発環境は [../development_environment.md](../development_environment.md) を参照。
