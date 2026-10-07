# MIA希少・適応4回版／非適応5回版の資料

4回版と5回版を、実ID・Team表記で確認するHTML資料です。4回版は第3回の209回答すべてから第4・5・6回の質問と所属配置の確定まで閲覧でき、`analysis/full_depth4_20261007/` の全53軌道証明書を圧縮したまま内蔵しています。

5回版の入口 **[MIA_5query_cards.html](MIA_5query_cards.html)** は、結果で人選を変えない **非適応の固定5回の提出** です。左の表に5提出×6チームの人数を並べ、横の6桁をチームごとの縦5桁へ読み替えます。選んだ列が右の11人それぞれの5桁の模様につながり、1人または2人を選んで模様を足すと所属レコードを確認できます。全体表とカードを同じ画面で表示し、小さい画面では表示を切り替えます。

従来の適応5回版は **[MIA_5query_adaptive.html](MIA_5query_adaptive.html)** に内容を変えず保存しています。適応版の最大回数は1人4回・2人5回。今回の非適応構成は1人・2人とも最大5回です。

## 開き方

1. [MIA_4query_tree.html](MIA_4query_tree.html) をGitHubの **Download raw file** で保存し、最新版のChrome・Edge・Safariで開きます。GitHubのファイル画面ではHTMLのソースが表示されます。
2. **具体例を1手進める** で、`2,494,800 → 40,428 → 631 → 20 → 1` を確認できます。**最後の20候補を見る** で第6回の直前へ移動できます。
3. 任意の回答行を選ぶと、その枝の次の質問が開きます。**1手戻る**、**最初に戻る**で経路を確認できます。人選・候補一覧・結果は実ID・Teamで表示します。
4. **平均点から回答を選ぶ** は、保存済みの216回答の点数表で復号した後、現在の枝にその回答が存在するかを検査します。
5. **現在の経路を保存** は回答codeと結果をJSONに、**残る全候補をCSVで保存** は現在の候補配置と予想回答をCSVに書き出します。

HTMLを開けば全データを利用でき、サーバー起動や証明書フォルダの選択は不要です。外部ライブラリの読み込みもありません。PDFのリンクも使う場合は、同じフォルダに [MIA_4query_tree_guide.pdf](MIA_4query_tree_guide.pdf) を保存してください。[配布ZIP](MIA_4query_visualizer_bundle.zip) を展開すると、HTML・PDF・説明書・サイズ記録・検証記録が揃います。

具体例は説明用の仮定経路で、実際の所属結果ではありません。Team 6・10・11・16・17・25の所属人数は1・2・1・2・1・2で、11候補のうち2候補はどのチームにも属しません。

## 非適応5回版と透明カードの使い方

1. 左の表は **1行＝1回の提出、1列＝1チーム** です。各提出の希少平均点から6チームの人数を読み取り、横の6桁を得ます。桁の順はTeam 6・10・11・16・17・25で、0・1・2は対象IDに含まれる所属者の人数です。提出の行を選ぶと、その行の人数列と説明用平均点を示します。
2. チームの列見出しを選びます。5回分を上から読むと、そのチームの縦5桁になります。右のカード欄のヘッダーにも **答え：チームの縦5桁** を表示します。所属1人では0・1、所属2人では0・1・2を使います。
3. 右のカードは **11人分** です。1枚に、その人が提出1〜5の対象に含まれるかを0・1で並べます。提出ごとに異なる色は使いません。カードを押して、比較する1人または2人を選びます。この操作はチームの結果を変えません。人数分を選び終えると重ねる操作を使えます。
4. **1提出ずつ照合** を押すと、その提出の0・1が選んだ1人または2人のカードから合計欄へ移動します。移動が終わってから、その提出だけの合計が現れます。まだ照合していない合計は空欄です。**選んだ模様を重ねる** は同じ足し算の動きを5提出分続けます。2人なら同じ提出の桁同士を足し、`0＋0＝0、0＋1＝1、1＋1＝2` と数えます。例：ID 47の `01011` ＋ ID 76の `10111` ＝ `11122`。ヘッダーの答えと1提出ずつ比較し、一致する桁に緑のチェック、不一致に×を表示します。5回とも合うと選んだレコードが緑になります。
5. **1提出ずつ照合／1提出戻す／やり直す／止める** で途中を確認できます。別の組を選ぶと人数列を保ったまま比較をやり直します。**一致するIDを選ぶ** は、固定された5桁全体を照合して合う1人または2人を選びます。55は2人の選び方の数で、表示するカードの枚数ではありません。
6. **所属1人／所属2人** のタブで人数を切り替えます。**正解レコード（説明用）** を変えた場合は、説明例そのものを変更して表とカードを更新します。他チームの例も同じ人が重複しないよう調整します。5回の対象IDは固定です。**例を戻す** で全6チームの初期例に戻ります。小さい画面では **全体の表／選んだチームのカード** を切り替え、動きを減らす設定では移動を省略します。
7. **詳しく見る** には候補数の計算と全列挙、固定コード、平均点の入力と照合があります。2人組は `11×10÷2＝55`、最初の提出の0・1・2人への分割は `15＋30＋10＝55`。初期Team 10の例は `55→30→15→7→2→1`、Team 6は `11→6→3→2→2→1`。平均点入力は説明用の正解選択と比較カードの選択とは別に保持します。この表は説明用の仮定です。

以前の5枚×55組の表示は **[MIA_5query_pairs.html](MIA_5query_pairs.html)** に内容を変えず保存しています。
以前の長い資料は **[MIA_5query_reference.html](MIA_5query_reference.html)** に内容を変えず保存しています。計算の元を一度に読む場合に使えます。

資料内の2,494,800は、6 Teamに同じ人を重複させず1・2・1・2・1・2人を割り当てる方法です。`11×45×8×21×5×6＝2,494,800`。216は平均点対応表の回答の種類で、`2³×3³＝216`。教材の11候補・55組とは数える対象が異なります。

今回の固定コードは **[nonadaptive_code.json](nonadaptive_code.json)** に保存しています。過去の資料には固定5bitコードへの言及がありますが、具体列を確認できなかったため、今回新しく構成した非適応案です。旧案の復元とは扱いません。

| 固定提出 | 対象の候補ID |
|---:|---|
| 1 | 76・169・264・462・675 |
| 2 | 47・90・115・264・675 |
| 3 | 76・90・264・383・495 |
| 4 | 47・76・90・169・304 |
| 5 | 47・76・115・462・495 |

11人の5桁の0・1コードがすべて異なり、異なる2人のコードを各桁で整数として足した55個の0・1・2回答列もすべて異なります。2人とも選ばれる回は2と数え、1には丸めません。各提出は5人なので、高側合計に合わせた既知非所属の追加数はTeam順に **1・2・2・6・5・7人** で全5回共通です。

全11＋55候補の一意性と、保存済み216平均点の一意復号を合成して6 Teamの追加5回保証を得ます。保証の方法は、4回版の全249万配置の再生検証と区別しています。既知非所属IDは正解人数hには寄与せず、高側合計の調整に使います。

従来の適応5回版の質問木は [追加5回の全候補確認.json](../検証資料/追加5回の全候補確認.json)、説明は [正式な適応5回案](../17_追加5回で個人特定できる理由.md) にあります。

## 保存質問と提出設定

4回版の質問は代表軌道の保存木に候補・チームの置換を適用し、5候補を超えるmaskを補集合に正規化したものです。回答codeも同じ正規化に合わせて変換しています。最後の質問には証明書上の子ノードがないため、残る配置から全回答を再計算し、各回答が1配置になることを確認して表示します。非適応5回版は上表の固定質問をそのまま使用します。

実ID表示では候補ID、追加する既知非所属ID、高側の合計人数を示します。選択候補と追加IDのconfidenceは **0.4**、その他の希少行は **0.1** です。高側の合計人数はTeam順に **6・7・7・11・10・12**。質問表の下にも設定を表示しています。

平均点の復号は保存点数表の最小間隔 `1/54000` の半分を距離の上限とし、境界・曖昧な丸め・現在の枝にない回答では停止します。使用する点数は、この保存質問と設定で得た点数に対応させてください。

## 実測サイズと形式の選択

サイズは10進単位のMBで示します。正確なbyte数・入力のSHA-256は [size_report.json](size_report.json) に保存しています。

| 対象 | byte数 | MB |
|---|---:|---:|
| 53証明書の元GZIP合計 | 1,080,682 | 1.081 |
| 53証明書の展開後JSON合計 | 16,487,312 | 16.487 |
| HTML内のGZIP Base64部分 | 1,440,984 | 1.441 |
| 4回版の単一HTML全体 | 1,500,969 | 1.501 |
| 4回版HTMLをGZIP圧縮した場合（参考実測） | 1,108,432 | 1.108 |
| 3ページの4回版ガイドPDF | 112,994 | 0.113 |

非適応5回版の最新サイズと入力ハッシュは [size_report_5query.json](size_report_5query.json)、55組の旧表示は [size_report_5query_pairs.json](size_report_5query_pairs.json)、旧資料は [size_report_5query_reference.json](size_report_5query_reference.json)、従来の適応版は [size_report_5query_adaptive.json](size_report_5query_adaptive.json) に保存しています。すべて単一HTMLで使えます。配布ZIPのファイル名は従来の `MIA_4query_visualizer_bundle.zip` を引き継ぎ、4回版・5回版・旧資料・PDF・説明書・固定コード・検証記録をまとめています。

HTML＋GZ分離ならBase64化の増分を減らせますが、ローカルファイルをそのまま開く際には証明書フォルダの指定や配信の仕組みが必要になります。今回は全53証明書を含めてもHTMLが約1.5 MBなので、**開くだけで使える単一HTML**を選びました。証明書をすべて同時に展開せず、選択した軌道だけを展開し、最大3軌道をキャッシュします。

GitHubは100 MiBを超える通常ファイルを拒否するため、このサイズは通常のGit管理で扱えます。[GitHub公式のファイルサイズ説明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。GZIPの展開には標準の [DecompressionStream](https://compression.spec.whatwg.org/#decompression-stream) を使用します。

## 検証

- 閲覧処理で全209回答・全2,494,800配置を再生し、対称移植、正規化、1,375,470質問ノード、終端の一意性、padding設定を照合しました。最終衝突は0です。[全配置の検証記録](../../tools/visualizer/validation_core.json)
- 生成HTMLの53内蔵GZIPと元証明書のbyte一致、すべてのGZIPの実展開、具体例と別軌道の第6回までの操作、戻る・リセット・平均点入力・JSON/CSV出力・欠落時の停止を確認しました。[操作処理の検証記録](../../tools/visualizer/validation_ui.json)
- HTMLの操作検証はNodeの模擬DOM環境で行いました。ブラウザのローカルファイル表示が実行環境の方針で制限されたため、実ブラウザでの目視確認は未実施です。PDFは全3ページを描画して目視確認しました。
- 5回版は、全11＋55候補・49質問ノード・66葉・深さ上限・padding・216平均点・説明経路を照合しています。[5回版の検証記録](../../tools/visualizer/validation_five_core.json)。操作処理の記録は [validation_five_ui.json](../../tools/visualizer/validation_five_ui.json) に保存しています。
- 従来の適応版2列の教材は、全11人・55組それぞれの5行の質問ID、前後の候補数、回答別の全列挙、候補カードの色と合成結果、途中で確定した行、例の切り替えを照合済みです。[適応教材の検証記録](../../tools/visualizer/validation_lesson_ui.json)
- 非適応版は固定提出、全11人・55組の整数回答列の一意性、330個の途中候補集合、padding、216平均点を照合済みです。[固定コードの検証記録](../../tools/visualizer/validation_fixed_core.json)。全66例×6段階の比較、11人分の全21,780個の表示ビット、全6チームの198通りの正解選択、横6桁・縦5桁・平均点・所属IDの一致、55通りの試行組に対する一致1組と不一致54組、結果を変えない比較選択、模様加算、全66例の1提出ずつの足し算アニメーション330段階、合計を移動後に1桁ずつ表示する処理、ヘッダーの答えの保持、アニメーションの中断、平均点照合1,080ケースを検証しています。記録は [validation_fixed_ui.json](../../tools/visualizer/validation_fixed_ui.json) にあります。55組の旧表示は [validation_fixed_pairs_ui.json](../../tools/visualizer/validation_fixed_pairs_ui.json)、旧資料は [validation_fixed_reference_ui.json](../../tools/visualizer/validation_fixed_reference_ui.json) で照合しています。位置は数式と模擬DOMで検証し、実ブラウザの描画確認とは区別しています。
- GitHubのmain `5abecbc7efd8cc626b8bd9d517dd477ad1ae9267` にある元証明書と、使用したローカル証明書のGit blob SHA一致も確認しました。

ChatGPTの旧HTML・PDF・ZIPは、この追加によって変更されません。このフォルダは検証済み証明書を元に作成した完全版です。既存の解析データ・提出資料・証明書への変更はありません。

## 再生成

元証明書は [../../analysis/full_depth4_20261007/](../../analysis/full_depth4_20261007/) にあります。MIA希少フォルダから実行します。

```sh
python3 tools/visualizer/build_visualizer.py
python3 tools/visualizer/build_five_visualizer.py
python3 tools/visualizer/build_fixed_visualizer.py
node tools/visualizer/test_tree_core.cjs --write-report
node tools/visualizer/test_viewer_ui.cjs --write-report
node tools/visualizer/test_five_core.cjs --write-report
node tools/visualizer/test_five_ui.cjs --write-report
node tools/visualizer/test_lesson_ui.cjs --write-report
node tools/visualizer/test_fixed_core.cjs --write-report
node tools/visualizer/test_fixed_ui.cjs --write-report
node tools/visualizer/test_fixed_pairs_ui.cjs --write-report
node tools/visualizer/test_fixed_reference_ui.cjs --write-report
python3 tools/visualizer/build_guide.py
python3 tools/visualizer/build_bundle.py
```

HTML・ZIP生成はPython標準ライブラリ、検証はNode.js 18以降を使います。PDFの再生成はPython 3.10以降、reportlab、pypdfが必要です。`build_guide.py --font /path/to/japanese.ttf` で日本語TTFを指定できます。検証の実行に追加のNodeパッケージは不要です。
