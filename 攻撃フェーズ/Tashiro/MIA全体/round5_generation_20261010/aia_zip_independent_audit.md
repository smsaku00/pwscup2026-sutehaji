# 第5回MIA + AIA/5_2 ZIP 独立検証

**PASS**。大会へ未提出。AIA・MIA・token原本を変更していない。

AIA原本は `b023ca627fb664ffb0f4b26f34ca6a4448b26418` の `攻撃フェーズ/AIA/5_2/結果/F_aia.csv`。Git blob `bade408752830d55a634bc1248bd49694c7664a1`、SHA256 `6f625e41a324ffb32ff2958c312a30ee2e06d3a3fac6504015443cb8f867d85c` と一致。

| 検証 | 結果 |
|---|---|
| AIA列順 | target_k, challenge_row_id, time_hat |
| AIA対象・行数 | MIAと同じ本戦22対象、各326行、全7172行 |
| 行キー | target_k / challenge_row_idは有限整数、重複0、全公開課題のID集合一致 |
| time_hat | 全有限・非負、9.5～11.0 |
| MIA | 検証済第5回原本のhash一致、1049行×22対象 |
| token | 本戦28配布folder出典、形式PASS、ZIP内byte一致 |
| 公式ZIP | 直下F_mia.csv / F_aia.csv / token.txt、CSV原本byte一致、CRC PASS |
| GitHub用ZIP | F_mia.csv / F_aia.csvのみ、同じCSV原本byte一致、CRC PASS |
| 公式検証器 | packaging / MIA logical / AIA logicalすべてPASS |

公開本戦mia_columnsと23課題を参照し、team_id=28、自B_selfのrecord_id集合を明示補完した。自己28を除く22対象すべてについて各326行の公開課題ID集合とAIA行を照合した。予備戦・本戦の所属正解は読んでいない。

token本文・token hashは表示も記録もしていない。公式ZIPにはtoken.txtを含むため、納品先はローカル/Libraryである。GitHubに掲載できるZIPは無token版であり、無token版だけでは公式提出要件を満たさない。

tokenの本戦28配布出典・形式・ZIP内一致を確認した。サーバー側token登録は非公開tokens.csvを持たないため検証していない。大会への送信・採点と推定精度評価は行っていない。

詳細は[JSON証跡](aia_zip_independent_audit.json)、再現は[独立監査スクリプト](aia_zip_independent_audit.py)を参照。
