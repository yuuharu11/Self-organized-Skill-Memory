# 計画：RoboCasa365 の atomic スキルメモリ

## 問い

atomic のデモから構築するラベルなしスキルメモリで、RoboCasa で微調整された π0.5 を atomic タスクで上回れるか。最終の比較は atomic_seen 18タスク×50試行（target split）の平均で、論文 Table 1 の multitask π0.5（Atomic 39.6%）と、同じ checkpoint を同じ条件で回したベースラインの両方と並べる。Composite-Unseen へのゼロショット転移は、その後の課題とする。

## 段階

1. **開発（6タスク）**：PickPlaceCounterToCabinet、PickPlaceSinkToCounter、OpenDrawer、CloseFridge、TurnOnSinkFaucet、TurnOffStove。メモリ・キー・大きさ・グリッパー・Expert の構成を決める。閉ループの判定は中位3タスク（CounterToCabinet、OpenDrawer、CloseFridge）×12試行を主に使う。
2. **拡張（18タスク）**：決めた構成のまま、18タスクの pretrain デモでメモリと各モデルを作り直し、target split の 18×50 で評価する。

## 学習の順番

メモリ構築 → クエリ（キー）の学習 → Expert の学習。後の段は前の段の出力に依存する（キーはメモリの型を教師にし、Expert はキーが選んだ下書きを入力にする）ので、前の段を決めてから進める。

## データと漏洩防止

- メモリ・キー・大きさヘッド・グリッパー頭・Expert は、pretrain の human デモのうちメモリ側の分割だけで作る。開発では各タスク100デモを 80/20 に分け、20 を保留にする。
- 保留デモの正解行動は採点と選択にだけ使い、学習・正規化統計・候補選択には使わない。自エピソードの単位は候補から除く。

## 比較

1. 同じ checkpoint の通常 π0.5（同じ seed）。
2. 論文の既報値。試行数・シーンの抽選・制御設定が一致しない場合は、直接の比較とは扱わない。
3. 参考：π0.5 の行動出力を選択に使う構成（R2 など）。学習済みタスクでは π0.5 が上限になるため、主張の根拠には使わない。

成功率、Wilson 95% 区間、タスク別件数、同じ seed との対応（McNemar 正確検定）を報告する。

## 計算量の実測

- 通常 π0.5 の評価：6タスク×50試行が、サーバー1台＋クライアント8〜12並列で約1時間。
- メモリだけの構成：1回の推論が約0.1秒（VLM の prefix 特徴だけを計算し、action expert は呼ばない）。
- データ：pretrain atomic は1タスク約500MB（100デモ）。18タスク分を取得済み。
