# 計画：RoboCasa365 のatomicスキルメモリ

## 問い

atomic の実演から構築するラベルなしスキルメモリが、target split の Composite-Unseen へゼロショット転移するか。

## データ分割と漏洩防止

- 学習・メモリ作成：target の atomic_seen 18タスクだけ。
- 主評価：target の composite_unseen 16タスクを各50試行。
- 補助：atomic_seen と composite_seen。
- Composite-Unseen の正解行動・区間ラベルは評価の採点に限り、学習、メモリ作成、候補選択、閾値選択には使わない。

## 比較

1. pi05_base を atomic_seen で微調整した baseline。
2. 同じ方策と評価予算で raw 区間メモリを利用。
3. RoboCasa公式の既報値。ただしcheckpoint、試行数、scene sampling、制御設定が一致しない場合は直接比較と扱わない。

成功率、Wilson 95%区間、タスク別件数を報告する。シミュレータ初期状態 seed とコード・checkpoint hash を保存する。

## 容量・計算の計画値

atomic target は18×500=9,000デモ、3カメラ映像を含む。空き容量は圧縮に依存するため0.5〜2TBを準備見積もりとし、実際の smoke/full download 後に更新する。学習を30k updatesで計画する場合、仮に1 update 0.3〜1.0秒なら約2.5〜8.3 GPU時間。評価800試行が1〜10分/試行なら直列で約13〜133時間（並列環境数で短縮）。いずれも実測前の粗い計画値であり、smoke速度で置き換える。ここでは学習も全評価もまだ実行していない。

## 準備済みと残作業

この段階では Python 3.11 venv、registry JSON、import/render smoke、smokeデータの取得・検査、demo replay配線までを確認する。full atomic取得、norm stats全量、pi05学習、メモリ作成、16×50評価は後続作業。
