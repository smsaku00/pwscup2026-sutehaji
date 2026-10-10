# 第5回MIAの生成・独立検証

[提出用MIAと確定方針](../../../MIA/5回目/README.md)。2026-10-10作成、**大会には未提出**。以前の第5回候補や過去12候補監査とは別の生成物です。

## 検証結果

- [独立最終監査](final_round5_independent_audit.md) / [JSON](final_round5_independent_audit.json)：PASS。
- [希少第3問の独立監査](rare_round5_final_audit.md) / [JSON](rare_round5_final_audit.json)：528セル一致。
- [公式検証器の取得元・原本一致](official_source_verification.json)：2026-10-10の公式mainと保存snapshotのGit blob一致。[config/schema依存の照合](official_dependencies_verification.json)も現行公式mainとbyte一致。
- [統計生成の再実行](methods/repeated_generation_validation.json)：選択CSV・区分表・内部配列がバイト一致。
- [統合の再実行](assembly_reproduction.json)：最終CSV・MIA配布ZIP・差分表・統合検証JSONがバイト一致。
- [追加指定AIA・完成ZIPの独立監査](aia_zip_independent_audit.md)：固定mainのAIA/5_2を変更せず組み合わせ、公式3ファイルZIPをローカル/Libraryへ納品。GitHubにはtokenなしMIA+AIA配布ZIPと記録を掲載。

1049行、22対象、IDと列の原本順序、有限[0,1]を確認しました。5対象以外の通常14875セルは第3回比差分0、Haruna本戦9の全1049セルは数値・文字列とも差分0。非希少の既知非共有150人×22＝3300セルは0。通常値域0.6003428571428571〜0.8999999999999999、希少は最大0.4です。

原raw値から4つのq7方式×22対象の公開非会員較正、q13方式×22対象のraw順位、全5方式×22対象の自身除外中央値補正とmin-rank変換を別実装で再計算し、誤差0でした。公式MIA logical検証にも、実際の本戦対象列と自分のBのID集合を渡してPASSしています。最終CSVのSHA256は `3e1f260ed7ab2f8beed57e6eb5b6e30d0c96388ac15c1e50ef0cfd1fe0f96c7a`。

第3回比は通常4366＋希少36＝4402セル変更、第4回比は通常13938＋希少36＝13974セル変更。[独立差分表](final_round5_independent_differences.csv)は提出側の44行の差分表と全一致。第4回通常部分の方式は第3回と異なるため、第4回比を5対象変更だけの効果とは解釈しません。

## 固定5bitと重複区分

第3回＝第1問、第4回＝第2問を保存原本1056希少セルで照合。今回の第3問のhigh候補は47,90,264,462,495。質問のpaddingも保存仕様に一致します。[希少期待値](rare_round5_expected.csv)、[区分ID](rare_round5_partition.json)、[履歴を含む照合](rare_round5_independent_audit.md)を保存しました。

既知非共有163人と希少24人には13人の重複があります。原仕様では重複13人を希少あり6対象の人数調整にも使い0.1/0.4を置くため、163人を一律0にすることは固定質問と両立しません。排他的な875/150/24を採用し、常時0は非希少150人です。Harunaは希少人数0なので全列固定と質問更新が両立します。公式score_miaを使った198の単体/2人組ケースと、216通りの応答符号の衝突なしも確認しました。

## 再現

元コード・入力・背景・seed・環境・順位変換は [方式生成のREADME](methods/README.md)、[生成manifest](methods/generated/generation_manifest.json)、[固定入力spec](methods/input_spec.json)を参照。原8コードは基準commit `1a5d47bba85711322994098b0de16df894acdf95` の回収原本に一致しています。背景3本各5143行と本戦B/C/Qの計50入力をGit blobおよびSHA256で検証しました。予備戦所属正解も本戦所属正解もfit・生成では読みません。

1. `methods/generate_selected.py`をルート`.venv/bin/python -B`で実行し、新規出力先へ選択5列・区分表・監査用NPZを作る。
2. `MIA/5回目/assemble_round5.py`へ、確認済み第3回CSV（SHA256 `6ca9604a8936fb47255ae0bd05c9f3958368f0e196bc2af258f8255cf39c08eb`）、第4回比較CSV（`45077532e5b5cf6109bf86ce67472fce0fac703584fa81a2f0027434fe581183`）、選択5列、区分表、`rare_round5_expected.csv`を渡し、新規出力先へ統合する。
3. `audit_fixed5bit_round5.py --candidate /path/F_mia.csv --report /path/new-report.json` と `audit_final_round5.py --candidate /path/F_mia.csv --component-zip /path/MIA_round5_components.zip --selected-ordinary /path/selected_confidence.csv --method-arrays /path/all_method_arrays.npz --method-manifest /path/generation_manifest.json --report /path/final-report.json` を実行する。ユーザー側checkoutが移動した場合は各`--local-root`を指定する。

各CLIの必須引数は`--help`で確認できます。独立監査器は生成器や統合器をimportしません。ローカル監査用NPZ、元入力、原本ログ、再実行コピー、tokenはGitHubへ再掲載しません。公開MIA配布ZIPはF_mia.csvのみで、公式の3ファイル攻撃提出ZIPとは区別します。実際の提出ZIPを作る場合は提出側の`package_submission.py`へ別途採用したAIAと配布tokenを指定します。

本戦の真のTPR・優位性・得点向上は未確認です。これらの検証は入力と実装・保存値の一致を示し、本戦性能を証明するものではありません。
