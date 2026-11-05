# 02 創発スキル：表現の作り方と軽量検証

## 問い

ラベルなしで、場面に依存しない「運動の型」を作れるか。作った型をメモリに足すと、新しい suite で行動誤差が下がるか。

## やったこと

凍結した π0.5 の VLM 特徴をキャッシュし、その上の小さな MLP で比べた（軽量版、オフラインの行動誤差のみ）。

1. 表現の比較 M0〜M6：連続 z、ベース＋残差（M1）、VQ、RVQ、辞書、FSQ、ART 型
2. 単位の作り方 M6：機能的距離で割り当て、まとまったら新しい単位として登録する。勾配なしで追加する
3. 取り出したスキルから行動を仕上げる検証

## 結論

- 表現は **M1（ベース b 12 次元＋残差 r 4 次元）** が最良。新しい suite 追加後の誤差は 0.117 / 0.092（Split A / B）。離散化（VQ・FSQ など）は表現力が足りない。
- 新しい運動の区間を当てる AUROC は 0.87。単位の追加率は新しい運動で約 7.5%、既知の運動で約 2%。
- 粒度 κ を、データだけから一意に決める規則はない。
- 環境での成功率、π0.5 本体での効果は未測定。この結果を受けて、型をフローの開始点に使う案（[03](../03_anchor_flow/README.md)）へ進んだ。

途中で見つかった訂正：特徴抽出で goal と spatial の suite 名が逆だった（修正済み）。言語ありの VLM 特徴は指示文が混ざるので、画像だけのクエリではない。

## ファイル

| ファイル | 中身 |
|---|---|
| [plan.md](plan.md) | 方針（決定事項と未決事項） |
| [verification.md](verification.md) | 最初の検証のアルゴリズムと判定基準 |
| [formation_study.md](formation_study.md) | スキル表現の作り方の比較（M0〜M6） |
| [construction_brief.md](construction_brief.md) | スキルの作り方の暫定方針（当時の方針。取り出しは softmax ブレンド） |
| [findings.md](findings.md) | 軽量検証の結果のまとめ |
| [representation_survey.md](representation_survey.md) | 追加可能なスキルメモリの表現学習（文献調査） |

実験スクリプトは [examples/emergent/light/](../../../skill_moe/skillnet/examples/emergent/light/README.md)。
