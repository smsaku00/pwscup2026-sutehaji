# 第5回希少MIA＋残り2問の独立監査

**4桁の画面表示だけでは40衝突群が残ります。解読には詳細HTMLの生の `mia.tpr_rare` を保存して使ってください。** 過去第1〜4回の生の値から得た1,296配置を前提に、将来3回は小数点5桁以上で衝突0でした。

**条件付きで、今回第5回＋予定第6・7回の固定質問により必ず特定できます。適応的な質問変更は不要です。** 全91分岐を完全列挙し、最悪残存配置数は48→12→1、最終観測列の衝突0です。

第5回は未提出・公式結果未確定として監査しました。第5〜7回の観測値は予測です。人数・候補集合・共有モデル・公式希少マスク・採点方式が正しく、残り2問もpadding込みの希少528セルを保持することが条件です。保存CSVと提出結果の対応は既存記録によるもので、サーバー入力ZIPとの直接hash照合は未実施です。

| 段階 | 観測分岐数 | 最悪残存数 |
|---|---:|---:|
| 第4回までの実観測 | 1 | 1,296 |
| 第5回予測 | 91 | 48 |
| 第6回まで予測 | 554 | 12 |
| 第7回まで予測 | 1,296 | 1 |

- [説明と4桁衝突の具体例](説明.md)
- [再実行スクリプト](audit.py)／[入力hashと所在](input_manifest.json)
- [全91分岐集計](round5_all_branches.csv)／[全1,296配置と観測](all_1296_placements_and_observations.csv)
- [最悪3分岐の完全列挙](worst_round5_branches.json)／[衝突0の機械検査](collision_check.json)
- [公式採点・原本照合の総合結果](audit_results.json)／[GitHub main原本8件照合](remote_originals_check.json)
- [表示精度別の衝突](rounding_precision.json)／[排他を外した補助検証](exclusivity_dependency.json)
- [複製ファイルのSHA-256・Git blob・掲載範囲](publication_manifest.json)

今回の確定MIA原本は[生成commitのF_mia.csv](https://github.com/smsaku00/pwscup2026-sutehaji/blob/27d40de61aeb025858038230802bd822381a773c/攻撃フェーズ/MIA/5回目/F_mia.csv)。SHA-256は `3e1f260ed7ab2f8beed57e6eb5b6e30d0c96388ac15c1e50ef0cfd1fe0f96c7a` です。

ここには監査成果を複製しました。`説明.md`・`source_review.json`等の「GitHub掲載未実施」は独立監査完了時の履歴として保存しています。監査元フォルダはローカルに保持し、生入力の `inputs/`、医療属性データ、認証情報、提出CSV・ZIPはこの掲載に含めていません。

ローカルでは原監査フォルダで `/Users/taisei1105/pws2026-local/.venv/bin/python -B audit.py` により再実行できます。GitHubから取得する場合は、`input_manifest.json` に従って本人の配布物・保存原本・公式コードを `inputs/` へ用意し、すべてのhashが一致する状態で同スクリプトを実行してください。生入力をGitHubへ追加する必要はありません。
