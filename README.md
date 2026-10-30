# Self-organized Skill Memory

ラベルなしに運動経験から自己組織化する「スキル」をメモリとして持ち、
Vision-Language-Action モデル（π0.5）の未学習の場面・タスクへの汎化を高める研究のコードと研究ノートです。
[SkillNet](https://github.com/VIPL-VSU/SkillNet)（ICML 2026）の公開コードをベースにしています。

> 研究は進行中です。現時点の結果は、凍結した π0.5 の特徴の上で小さなモデルを学習する軽量版での、
> オフラインの行動誤差が中心です。環境での成功率はまだ確定していません。
> 最新の状況は [docs/research/skill_memory_findings.md](docs/research/skill_memory_findings.md) を参照してください。

## 初めて読む方・再現する方へ

**別の環境での手順は [再現ガイド](docs/reproduction.md) から始めてください。**
新規の環境構築、公開重みの取得、実行コマンド、出力の確認先、研究実験に必要な追加資産をまとめています。

| 確認したいこと | 読む場所 |
| --- | --- |
| この研究の仮説と測定結果 | [研究ノートの入口](docs/research/README.md)、[これまでの結果](docs/research/skill_memory_findings.md) |
| 別環境でコードを動かす・公開重みを評価する | [再現ガイド](docs/reproduction.md) |
| 各研究スクリプトの役割 | [軽量実験の一覧](skill_moe/skillnet/examples/emergent/light/README.md)、[中間行動実験の一覧](skill_moe/skillnet/examples/emergent/light/intermediate/README.md) |
| ベースの SkillNet の学習・データ処理 | [Quick Start](docs/quick_start.md)、[学習と評価](docs/training_and_evaluation.md) |

コードと小さな設定・注釈は Git に含まれます。データセット、重み、特徴キャッシュ、実験出力は別途用意します。
研究ノートの `.skillnet-state/` 以下へのリンクは測定時のローカル記録で、公開ダウンロード先ではありません。
公開重みの評価と、研究ノートの数値を同じ条件で再現するための要件は、再現ガイドでそれぞれ説明しています。

## 検証する仮説

1. **スキル = 場面に依存しない運動の型。** タスク・言語のラベルなしに運動経験から自己組織化し、個数は事前に決めない。
   場面との対応はメモリ（鍵 = 観測の埋め込み、値 = 運動の型）が受け持つ。
2. **継続学習に対応する。** 新しい運動は、勾配による更新なしにメモリへ単位として追加される。
3. **スキルを外部の条件として与えない。** 条件は画像・サブタスク・ロボット状態の埋め込みだけ。
4. **デノイズは、根本的な運動を状況に合わせて最適化する過程である。**
   メモリから取り出した運動の型の混合を、フローマッチングのデノイズの開始点にする。

```
b = Enc(A, s)                                  行動チャンク A と状態 s から運動のコード b
w_k(o) = softmax_k( cos(E(o), κ_k) / τ )       観測 o の埋め込みでメモリの単位 k を重み付け
x_{t0} = (1 − t0)·Dec(b') + t0·ε               取り出した運動の型 b' から t0 でデノイズを開始
```

詳細は [docs/research/anchor_flow_skill_memory.md](docs/research/anchor_flow_skill_memory.md)。

## リポジトリ構成

SkillNet 由来の構成（`skill_moe/`, `data_process/` など）に、次を追加しています。

| パス | 内容 |
| --- | --- |
| `docs/research/` | 研究ノート（方針・判定基準・結果・関連研究）。入口は [docs/research/README.md](docs/research/README.md) |
| `skill_moe/skillnet/examples/emergent/` | 創発スキルの軽量検証（スキル表現の比較 M0〜M6、運動の型から始めるデノイズの S1） |
| `skill_moe/skillnet/examples/memory_pilot/` | 検索メモリの事前判定（G0〜G3）と、LIBERO 閉ループでのデノイズ開始点の検証（S2） |
| `skill_moe/skillnet/src/openpi/models/pi0_anchor_flow.py` ほか | デノイズ開始点の注入（anchor flow）とスキル潜在を持つ π0 の変種 |
| `env/` | 現在のコンテナ用の環境設定・スモークテスト・依存の固定リスト |
| `tests/` | 追加コードのテスト |

## コードを取得する

```bash
git clone https://github.com/yuuharu11/Self-organized-Skill-Memory.git
cd Self-organized-Skill-Memory
git rev-parse HEAD
```

この案内を含む **PR #2 をマージ前に確認する場合**は、取得後に次を実行します。
共有・比較の際は、ブランチ名と併せて `git rev-parse HEAD` のコミット ID を記録してください。

```bash
git fetch origin pull/2/head
git switch --detach FETCH_HEAD
git rev-parse HEAD
```

続いて [再現ガイドの環境構築](docs/reproduction.md#2-別環境でのセットアップ) へ進んでください。
`env/activate.sh` はインストール済みの環境を有効化するスクリプトです。新規 clone には依存の導入が必要です。
元のコンテナの検証記録は [docs/environment_validation.md](docs/environment_validation.md) にあります。

## ベースの SkillNet

SkillNet の学習・評価（LIBERO-40/90、LIBERO-Skill、RoboTwin few-shot）の手順は、元の README を
[docs/skillnet.md](docs/skillnet.md) に移して残しています。データ処理・チェックポイント・評価の詳細は
[docs/quick_start.md](docs/quick_start.md) と [docs/training_and_evaluation.md](docs/training_and_evaluation.md) にあります。

SkillNet を使う場合は、次を引用してください。

```bibtex
@inproceedings{xie2026skillnet,
  title={SkillNet: Hierarchical Skill Modeling for Compositional Generalization in Vision-Language Action Models},
  author={Xie, Senwei and Zhang, Yuntian and Tan, Zhenzhou and Wang, Ruiping and Wang, Pengwei and Zhang, Shanghang and Chen, Xilin},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```

## License

特に記載のない限り Apache-2.0 です。SkillNet および第三者のコンポーネントは元のライセンスに従います。
`NOTICE.md` と `skill_moe/skillnet/third_party/` 以下のライセンスファイルを参照してください。
