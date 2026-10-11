# 全配布人数条件での MIA 希少方式の検証

人数が既知になった後の個人特定を検証した証明 bundle です。候補11人、対象22チーム、各対象0〜2人、候補の所属先は高々1チーム、未割当を認めます。候補と既知非会員13人を合わせた24行に二値の高低回答を提出し、人数が正の対象だけの**丸め前の希少TPR平均**を観測するモデルです。【根拠：組合せ全列挙・有理数による完全検算・保存決定木の全状態再生】

| 問い方 | 42人数型の結論 |
|---|---|
| 今回採用した共通の固定5問 | 全42型で保証。全ゼロ型は自明 |
| 全対象で共通の候補符号を使う固定4問 | 14型可能、28型不可能。完全分類 |
| 対象ごとに候補符号を変える固定4問 | 18型可能、20型不可能、4型未判定 |
| 回答に応じて次の問いを変える適応4問 | 23型で可能と分かっている、4型不可能、15型未判定 |

適応4問の23型は、固定4問で解ける18型、旧 `(3,3)` 型、追加証明した `(4,3), (2,4), (0,5), (1,5)` 型です。固定で解ける18型の適応探索は実施していません。適応の全42型完全分類は未完了です。1チームだけに2人所属する `(0,1)` 型でも適応4問は不可能なため、**この二値回答モデルで全配布条件を保証する個人特定回数は最小5問**です。

ここで `(a,b)` は人数1の対象が `a` チーム、人数2の対象が `b` チームという意味です。`a+2b<=11` の42型で、各対象の名前を入れ替えても、問い・M設定・復号した所属先を一緒に入れ替えることで証明を移せます。型ごとの探索に22チームの全ラベル配置を区別する必要はありません。正確な各型の判定は [CSV](classification/types.csv)、[総合JSON](classification/results.json) にあります。

## 入口と証明の位置

- [説明](説明.md)：前提・方法・保証範囲と限界。
- [公開 bundle manifest](bundle_manifest.json)：再現コードが使う相対パスと型別証明の配置。
- [固定5問](fixed5/results.json)、[M設定・型別件数](fixed5/types.csv)、[全13,103個の正人数型平均回答と全ゼロ回答](fixed5/exact_answers.json)、[独立監査](fixed5/independent_audit.json)。
- [固定4問の下界と共通符号の完全分類](fixed4/bounds.json)、[対象別符号のSAT証拠3件](fixed4/search.json)、[追加SAT証拠1件](fixed4/optimized_b4_m11.json)、[独立監査](fixed4/independent_audit.json)。4件の追加符号と共通符号で解ける14型を合わせて、対象別固定4問の18型を保証します。
- [追加4型の適応4問集計](adaptive4/results.json)、[3型の平均回答全状態再生](adaptive4/mean_replay.json)、[`(4,3)` 型の全状態再生](adaptive4/a4b3/replay.json)。4型合計9,979,200個の個人所属配置を具体的に再生し、衝突0、最大4問を確認しています。
- [旧 `(3,3)` 型の独立再検証](adaptive4/a3b3/replay.json)。旧53代表証明は既存の [full_depth4_20261007](../full_depth4_20261007/) を参照します。
- [適応不可能アンカー](adaptive4/bounds/)：旧ソルバーの確定証明原本。基本監査は候補9人の単一ペアを独立に完全再計算し、他対象の所属を固定した部分問題への帰着で4型の不可能性を確認します。`anchor_b2` 専用の56初問枝探索は再実行しません。詳しい再計算範囲は [再現コードのREADME](../../experiments/universal_distribution_20261011/README.md) を参照してください。
- [原本と公開ファイルのSHA対応](provenance.json)、[大容量列挙データの仕様とSHA](release_attachment.json)。

新しい3型の決定木は、`adaptive4/a1b5/`, `adaptive4/a2b4/`, `adaptive4/a0b5/` に各 `manifest.json.gz` と代表番号ごとの `certificate.json.gz` を保存しています。代表数は6・14・6の計26です。代表木を対象の置換で全到達初問枝へ輸送します。途中の探索記録と重複した巨大な全体木は掲載していません。

## 全列挙の意味

22チームを区別した人数ベクトル `c∈{0,1,2}^22, sum(c)<=11` は82,995,960個です。この保存データは実際に全生成し、独立監査で全件を読み、ビット範囲・非重複・人数・件数・昇順・SHAを検算しました。うち全員割当の `sum(c)=11` は49,402,850個です。記録は [列挙metadata](fixed5/enumeration_manifest.json) と [監査](fixed5/independent_audit.json) に保存しています。

固定5問については、全人数ベクトルごとに全個人所属を直接再生したわけではありません。各人数型の平均回答が各対象の人数回答を一意に復号でき、今回の5問の符号が11個の単独候補・55個の候補ペアを一意に区別できることを合成して、すべての整合する個人所属を保証します。

列挙payloadは131,304,256 bytes（約125.22 MiB）のため、このGit treeに含めません。外部添付または再生成で用意した場合は、公開監査へ `--enumeration-file PATH` で渡せます。候補・属性・競技配布データは不要です。生成形式とraw/gzip SHAは [添付metadata](release_attachment.json) にあります。

GitHub ReleaseではASCII名 [all-count-patterns.uint64.gz](https://github.com/smsaku00/pwscup2026-sutehaji/releases/download/mia-universal-distribution-20261011/all-count-patterns.uint64.gz) で取得できます。取得したファイルのパスを `audit.py --enumeration-file PATH` に渡してください。ローカル再生成時のファイル名は `全人数パターン.uint64.gz` です。

再生成はrepository rootで次のように実行します。出力先は新しいフォルダを指定してください。各42blockと全体のraw SHAを照合し、gzipの一致は実行環境の圧縮実装も含めた照合結果として記録します。

```bash
.venv/bin/python -B 攻撃フェーズ/Tashiro/MIA希少/experiments/universal_distribution_20261011/enumerate.py --output-dir /tmp/mia-enumeration-new
.venv/bin/python -B 攻撃フェーズ/Tashiro/MIA希少/experiments/universal_distribution_20261011/audit.py --enumeration-file /tmp/mia-enumeration-new/全人数パターン.uint64.gz
```

## 再現と原本の扱い

[公開用の独立監査コード](../../experiments/universal_distribution_20261011/README.md) は、このbundleの相対パスと既存の旧53代表証明を使います。元の分析コードや生成ソルバーをimportしません。既定の実行は全状態再生を含みます。

元の証明JSON・CSV・gzip45ファイルは内容を変更せずコピーし、SHA-256を記録しました。JSON中に残るローカルパス、対象ラベル、当時のPython情報は出典の追跡用です。公開監査はそれらのローカルパスに依存しません。新しく追加したmetadataと平均回答表は候補スロット `0..10`、対象スロット `0..21` のみで扱います。

人数把握の最初2問、13人の非会員認定、Cや公開属性による前処理が任意の配布条件で成功することまでは証明していません。24行・候補11人・非会員13人・人数既知の条件が整ってからの保証です。任意の多値信頼度回答、画面上の丸めた数値だけでの復号、実際の配布確率もこの検証の対象外です。
