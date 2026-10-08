# MIA希少 scripts — コードの対応時期と実行条件

> **【2026-10-08追記】** 下記の`fast_solver_20261007`は適応4回案の研究用で、実提出の固定5bitとは別です。実採用方式の生成・監査・読取器は[第3回MIAの共通原本](../../../MIA/20261008_第3回MIA_固定5bitと非会員較正/README.md)にあります。

## 1 適応4回案の探索コードと独立checker

- 2026年10月7日の探索実装：[fast_solver_20261007のREADME](../experiments/fast_solver_20261007/README.md)
- 保存場所：`experiments/fast_solver_20261007/`
- 探索：`solver.py`・`constrained_search.py`・`solve_all_orbits.py` など
- 独立checker：`verify.py`・`verify_all_orbits.py`・`verify_score_encoding.py`、探索コードとの分離
- 完成証明書：[analysisの索引](../analysis/README.md)
- 次の人選・paddingの参照：[4回案の実行手順](../docs/18_追加4回の進め方と完遂保証.md)
- 実行環境：PWS作業ルートのHomebrew Python 3.12・`.venv/bin/python`

## 2 2026年9月28日の初期242組分析

- この `scripts/` の4ファイル：初期候補の比較・検証・資料生成に対応
- 現在の全決定木探索：上記 `experiments/` が参照先

| コード | 役割 |
| --- | --- |
| [分析.py](分析.py) | 初期候補の距離・固有性等を計算 |
| [検証.py](検証.py) | 初期分析の独立検証 |
| [共通設定.py](共通設定.py) | 引数と入力ハッシュの照合 |
| [レポート.py](レポート.py) | 集計・図・説明を生成 |

- 対応する資料：[当時の説明](../archive/20260928_初期候補分析/説明.md)／[保存結果](../analysis/README.md)

## 3 初期分析を再計算する際の配置

- 保存コードの前提：`__file__` の隣に入力ハッシュ・候補判断・説明テンプレートのある配置
- 現在のGitHub配置：役割別の資料分離／この `scripts/` のままの実行は未検証
- 初期分析の再計算：新しい作業フォルダに下記の保存ファイルを同じ階層で配置、当時のコード・入力ハッシュを確認
- 上表の4コード
- [入力ハッシュ.json](../analysis/入力ハッシュ.json)
- [候補選定.json](../analysis/候補選定.json)
- [説明テンプレート.md](../archive/20260928_初期候補分析/説明テンプレート.md)
- 自チーム28の `B_self.csv`、本戦 `A_bg.csv`、固定版の攻撃配布入力は別途必要
- 実行環境：PWS作業ルートのHomebrew Python 3.12・`.venv/bin/python`／出力先：新しい空フォルダ
- 説明テンプレートのリンクはGitHub上の保存資料へ合わせてあり、再生成資料を別の場所で配布する際は配置に合わせて確認が必要
- 初期242組分析のコード・入力・結果：保存内容を維持、今回の再実行は未実施
- 再現できたことを新たに確認した記録とは別
- 決定木探索の保存範囲は [docsの索引](../docs/README.md) と [analysisの索引](../analysis/README.md) を参照
