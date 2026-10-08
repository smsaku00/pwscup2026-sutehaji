# MIA希少 analysis — 完成証明書と保存結果の索引

> **【2026-10-08の位置付け】** 以下の完成証明書・`inspect_strategy.py`は**適応4回の比較方式**に対応し、第3回に実提出した非適応・固定5bitの質問列には対応しません。第3回の実回答・設定・監査・結果読取器は[共通MIA](../../../MIA/20261008_第3回MIA_固定5bitと非会員較正/README.md)を参照してください。旧証明書は変更せず保存します。

- 更新日：2026年10月7日
- 説明の入口：[MIA希少のREADME](../README.md)、[docsの索引](../docs/README.md)
- 現在の実行用質問木：`full_depth4_20261007/`

## 1 追加4回の完成証明

- [最終集計](full_depth4_20261007/summary.json)：SAT、全2,494,800状態、209到達q3枝、53代表軌道
- [独立全状態audit](full_depth4_20261007/independent_full_audit.json)：全state replay、衝突0、最大追加depth4
- [移植対応と証明書一覧](full_depth4_20261007/manifest.json)：全209枝と53代表の対応、各代表のq4・q5・q6 witness
- [全209枝の第4回masks](full_depth4_20261007/q4_masks_normalized.csv)／[軌道別集計](full_depth4_20261007/orbit_summary.json)
- q4後：34,902枝、最大bucket1,338
- q5後：1,340,358セル、最大cell44、全セルでq6のexact witnessを構成・検証済み
- [平均点の共通設定](full_depth4_20261007/universal_score_encoding.json)／[padding規則](full_depth4_20261007/padding_rule.json)／[公式採点との照合](full_depth4_20261007/score_encoding_check.json)
- [code30の具体例](full_depth4_20261007/example_witness_code30.json)
- 詳しい保証：[16 全状態証明](../docs/16_追加4回完遂の全状態証明.md)
- 実行時の参照：[18 4回案の説明](../docs/18_追加4回の進め方と完遂保証.md)、[質問参照ツール](../experiments/fast_solver_20261007/inspect_strategy.py)
- `initial_search_progress.json.gz`：途中のUNKNOWNを含む履歴、最終判定は完成証明書・audit・最終集計

## 2 code30・246・270の完遂と探索履歴

- [3枝の集計](code30_continuation_20261007/summary.json)／[全状態replay](code30_continuation_20261007/full_replay.json)
- [15 完遂報告](../docs/15_q3軌道30_246_270_6回完遂.md)：各40,428状態、q4到達208枝、q4最大bucket899、q5最大cell20、q6衝突0
- 制限付き探索の未発見：UNKNOWN、対象を固定した完全列挙によるUNSATとは区別

### 完成前の保存結果

| 保存結果 | 対応する説明・範囲 |
| --- | --- |
| [6回完遂探索の上位枝](6回完遂探索_上位枝.csv) | [06 初期探索](../docs/06_6回完遂探索_20261007.md) |
| [bucket 117の不可能コア](bucket117_unsat_core.csv) | [07 不可能証明](../docs/07_bucket117_残り2回不可能証明.md)／棄却した旧第4回案 |
| [新第4回の上位10枝screen](新第4回_pairwise_screen_top10.csv)／[上位10枝の第5回分離](新第4回_top10_q5完全分離.csv) | [08 中間結果](../docs/08_新第4回_top10完全決定木.md)／この案全体は [09](../docs/09_新第4回_215枝解決_1枝不可能.md) で棄却 |
| [q3 code 274の第5回質問表](q3_274_q5_map.json) | [11 軌道274/282/354](../docs/11_q3次点軌道274_6回完遂.md) |
| [q3 code 273の第5回質問表](q3_273_q5_map.json) | [12 軌道273](../docs/12_q3軌道273_6回完遂.md) |

- 上表：一部軌道から順に構成した当時の結果
- 軌道273の旧q5表：1枝を再構成、最終質問木とセル数は完成証明書・auditが参照先

## 2026-09-28の初期候補分析

- `集計/`・`図/` と以下の記録は初期の242組比較に対応
- 10月5日・6日の提出で導いた人数や、最新の決定木探索の検証結果とは別
- [候補242組](集計/候補242組.csv)／[チーム別候補](集計/チーム別候補.csv)／[希少24行の区分](集計/希少24行の区分.csv)
- [候補比較の図](図/候補比較.png)／[当時の暫定判断](候補選定.json)
- [入力ハッシュ](入力ハッシュ.json)／[実行記録](実行記録.json)／[検証結果](検証結果.json)
- [当時の説明と列定義](../archive/20260928_初期候補分析/説明.md)／[対応コードと再現条件](../scripts/README.md)
- 提出CSVと採点原本は [共通MIA](../../../MIA/README.md) に保存
