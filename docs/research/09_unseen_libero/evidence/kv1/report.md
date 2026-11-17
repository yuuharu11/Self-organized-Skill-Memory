# KV1: 実装状況

## 結果

**未完了：最終層の既存キャッシュとのパリティゲートで停止。** 学習やEvalへ進める根拠がないため、性能値や順位を作っていない。

18層の prefix K/V を `[18, 2, 532, 256]` fp16 で保存する再開可能な抽出器、K/V 読者（KV-L、KV-H、KV-LH、選択層・層別attention対応）の実装とCPUテストを用意した。テストは8件成功。KV-L90X の固定78タスク漏洩検査も追加した。テスト結果は `pytest -q tests/unseen_libero/test_key_kv_cache.py` で8 passed。

実データの確認では、行 `(task=3, episode=6, frame=0)` の画像2枚・prompt ID・maskは run-01 の行と完全一致し、state の最大差は `1.34e-8`。今回の直接 prefix 経路と `core.prefix_features` の比較は通った。一方、既存 run-01 token cache との比較は最大絶対差 `0.502685546875` で失敗した（最初の報告位置 `(domain=0, task=0, episode=58, frame=0)`）。この差の原因はまだ特定できていない。既存キャッシュの出所 manifest は checkpoint path を示すが、重み内容のハッシュや抽出コード版を固定していないため、同一重み・同一実装だったことまでは確認できない。

そのため、全量 K/V の抽出、モデル学習、Dev 選択、Eval、比較表は実行していない。途中の smoke memmap は `smoke/features/` にあり、`complete=false` / `done=false` のまま保持している。学習開始・再利用可能な完成キャッシュとは扱わない。

## 継続に必要な確認

1. 既存 token cache を生成したコードと checkpoint の実体を特定し、固定重みで同一行の最終層出力を再生成する。
2. `MemoryPi05.prefix_features` と今回の直接 `PaliGemma.llm` 経路の一致に加えて、run-01 token cache と完全一致することを確認する。
3. パリティが解消してから全行のキャッシュを作り、KV/H/latent 条件を学習し、Dev の top-1 腕MSEで選択して初めてEvalを一度採点する。

`/work/docs/memory_pilot_gates.md` はこの checkout に存在しなかった。既存ジョブは停止・変更していない。

## 成果物

- 実装: `skill_moe/skillnet/examples/memory_pilot/unseen_libero/key_kv_cache.py`
- テスト: `tests/unseen_libero/test_key_kv_cache.py`
- 詳細機械可読結果: `results.json`
