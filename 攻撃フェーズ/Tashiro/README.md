# Tashiro — 主担当はMIA希少

**田代の主担当はMIA希少です。**  
MIA全体・AIA全体・AIA希少は、田代が補助的に検討した分析・担当者向けの提案です。担当の引継ぎや、チームとしての採用決定を意味しません。

## まず読む場所

| 区分 | テーマ | 入口 | 状態 |
| --- | --- | --- | --- |
| **主担当** | **MIA希少** | [MIA希少/README.md](MIA希少/README.md) | 個人特定の決定木を探索中 |
| 補助分析 | MIA全体 | [MIA全体/README.md](MIA全体/README.md) | 提案候補あり |
| 補助分析 | AIA全体 | [AIA全体/README.md](AIA全体/README.md) | 提案候補あり |
| 補助分析 | AIA希少 | [AIA希少/README.md](AIA希少/README.md) | 要本戦検証 |

MIA第1回・第2回の**提出原本・採点結果・人数導出**は、チーム共通で追えるように [攻撃フェーズ/MIA](../MIA/README.md) を正とします。

## フォルダ構成

```text
Tashiro/
├── MIA希少/                 # 主担当
│   ├── proposal/
│   ├── rounds/
│   ├── docs/
│   ├── analysis/
│   ├── scripts/
│   ├── reference/
│   └── archive/
├── MIA全体/                 # 補助分析
│   ├── proposal/
│   └── experiments/20261007/
├── AIA全体/                 # 補助分析
│   ├── proposal/
│   └── experiments/20261007/
├── AIA希少/                 # 補助分析
│   ├── proposal/
│   └── experiments/20261007/
├── notes/                   # 過去メモ・整理記録
└── archive/                 # 旧計画
```

**日付付きの大型実験も各テーマの `experiments/` に移しました。**  
通常は各テーマのREADMEを読み、詳細なコード・CSV・JSONが必要なときだけ `experiments/20261007/` へ進んでください。

## MIAの提出履歴

実際の提出回は共通MIAに残します。

- [第1回：希少同点人数探索](../MIA/20261005_MIA希少_第1回同点データ/README.md)
- [第2回：seo-02と希少同点](../MIA/20261006_MIA第2回_seo02と希少同点/README.md)
- [第1・2回からの希少人数導出](../MIA/第1回と第2回からの希少MIA人数の導出.md)

`Tashiro/MIA希少/` は、この共通原本を参照しながら第3回以降の個人特定を進める作業領域です。

## AIAについて

共通 `攻撃フェーズ/AIA/` にはAIA担当者側の `一回目/`・`第2回/` を残しています。田代が追加していた独自の案内・比較方針は [notes/AIA_20261006/](notes/AIA_20261006/README.md) に移しています。

担当外の共通領域へ、田代の補助提案や独自方針を追加・上書きしない方針です。

## 過去資料

- [notes/](notes/README.md)：過去メモ、比較案、整理記録
- [archive/](archive/README.md)：現在は採用していない旧計画

整理履歴は [notes/整理記録_20261007.md](notes/整理記録_20261007.md) に残しています。
