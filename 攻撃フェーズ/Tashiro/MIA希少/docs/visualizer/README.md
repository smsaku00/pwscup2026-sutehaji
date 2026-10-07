# MIA希少・4回版／5回版の決定木資料

4回版と5回版を、実ID・Team表記で確認するHTML資料です。4回版は第3回の209回答すべてから第4・5・6回の質問と所属配置の確定まで閲覧でき、`analysis/full_depth4_20261007/` の全53軌道証明書を圧縮したまま内蔵しています。

5回版の入口は **[MIA_5query_cards.html](MIA_5query_cards.html)** です。所属候補の印と質問の選択枠を重ねる透明カードの説明、6 Team同時の第3〜7回の人選・padding・平均点、回答に合う候補の絞り込みを確認できます。1人用の全11候補・2人用の全55組の質問木を内蔵しています。

## 開き方

1. [MIA_4query_tree.html](MIA_4query_tree.html) をGitHubの **Download raw file** で保存し、最新版のChrome・Edge・Safariで開きます。GitHubのファイル画面ではHTMLのソースが表示されます。
2. **具体例を1手進める** で、`2,494,800 → 40,428 → 631 → 20 → 1` を確認できます。**最後の20候補を見る** で第6回の直前へ移動できます。
3. 任意の回答行を選ぶと、その枝の次の質問が開きます。**1手戻る**、**最初に戻る**で経路を確認できます。人選・候補一覧・結果は実ID・Teamで表示します。
4. **平均点から回答を選ぶ** は、保存済みの216回答の点数表で復号した後、現在の枝にその回答が存在するかを検査します。
5. **現在の経路を保存** は回答codeと結果をJSONに、**残る全候補をCSVで保存** は現在の候補配置と予想回答をCSVに書き出します。

HTMLを開けば全データを利用でき、サーバー起動や証明書フォルダの選択は不要です。外部ライブラリの読み込みもありません。PDFのリンクも使う場合は、同じフォルダに [MIA_4query_tree_guide.pdf](MIA_4query_tree_guide.pdf) を保存してください。[配布ZIP](MIA_4query_visualizer_bundle.zip) を展開すると、HTML・PDF・説明書・サイズ記録・検証記録が揃います。

具体例は説明用の仮定経路で、実際の所属結果ではありません。Team 6・10・11・16・17・25の所属人数は1・2・1・2・1・2で、11候補のうち2候補はどのチームにも属しません。

## 5回版と透明カードの使い方

1. **具体例を1手進める** で、6 Teamを並行して調べる第3〜7回の例を進めます。Team10は `55→28→13→5→2→1組` と絞れます。
2. **図で見るTeam** で対象を変えます。候補カードを選ぶと、所属候補の印が質問の枠に何個入るかを確認できます。灰色のカードは過去の回答で除外された候補です。
3. **回答済みの透明シート** は、「この選択IDの中にh人いる」という過去の条件です。重ねたすべての条件に合うカードだけを残します。2人のTeamでは、個人の集合だけでなく2人組を候補として保持します。
4. 各Teamの0・1・2人の回答を選んで **この回答で次へ** を押すと、任意の枝へ進めます。保存木の分岐と、同じIDが複数Teamに所属しない条件を満たせない回答では停止します。
5. **平均点から6 Teamの回答を読む** と **現在の経路を保存** も使えます。各Teamの候補数は独立に数えており、その積を合法な全体配置数とは扱いません。

質問木は [追加5回の全候補確認.json](../検証資料/追加5回の全候補確認.json) そのものです。1人用は最大4回、2人用は最大5回。全11＋55候補の確認と216平均点の一意復号を合成して6 Teamの追加5回保証を得ます。保証の方法は、4回版の全249万配置の再生検証と区別しています。[正式な5回案の説明](../17_追加5回で個人特定できる理由.md)

5回版の説明経路は保存済みホワイトボード例の仮定配置を表示したものですが、平均点は今回の固定高側合計で再計算しています。過去資料のランダムpaddingの点数を流用していません。既知非所属IDは透明カードに所属の印がないため、hの人数には寄与せず、高側合計の調整に使います。

## 保存質問と提出設定

質問は代表軌道の保存木に候補・チームの置換を適用し、5候補を超えるmaskを補集合に正規化したものです。回答codeも同じ正規化に合わせて変換しています。最後の質問には証明書上の子ノードがないため、残る配置から全回答を再計算し、各回答が1配置になることを確認して表示します。

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

5回版の最新サイズと入力ハッシュは [size_report_5query.json](size_report_5query.json) に保存しています。両版とも単一HTMLで使えます。配布ZIPのファイル名は従来の `MIA_4query_visualizer_bundle.zip` を引き継ぎ、4回版・5回版・4回版PDF・説明書・検証記録をまとめています。

HTML＋GZ分離ならBase64化の増分を減らせますが、ローカルファイルをそのまま開く際には証明書フォルダの指定や配信の仕組みが必要になります。今回は全53証明書を含めてもHTMLが約1.5 MBなので、**開くだけで使える単一HTML**を選びました。証明書をすべて同時に展開せず、選択した軌道だけを展開し、最大3軌道をキャッシュします。

GitHubは100 MiBを超える通常ファイルを拒否するため、このサイズは通常のGit管理で扱えます。[GitHub公式のファイルサイズ説明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。GZIPの展開には標準の [DecompressionStream](https://compression.spec.whatwg.org/#decompression-stream) を使用します。

## 検証

- 閲覧処理で全209回答・全2,494,800配置を再生し、対称移植、正規化、1,375,470質問ノード、終端の一意性、padding設定を照合しました。最終衝突は0です。[全配置の検証記録](../../tools/visualizer/validation_core.json)
- 生成HTMLの53内蔵GZIPと元証明書のbyte一致、すべてのGZIPの実展開、具体例と別軌道の第6回までの操作、戻る・リセット・平均点入力・JSON/CSV出力・欠落時の停止を確認しました。[操作処理の検証記録](../../tools/visualizer/validation_ui.json)
- HTMLの操作検証はNodeの模擬DOM環境で行いました。ブラウザのローカルファイル表示が実行環境の方針で制限されたため、実ブラウザでの目視確認は未実施です。PDFは全3ページを描画して目視確認しました。
- 5回版は、全11＋55候補・49質問ノード・66葉・深さ上限・padding・216平均点・説明経路を照合しています。[5回版の検証記録](../../tools/visualizer/validation_five_core.json)。操作処理の記録は [validation_five_ui.json](../../tools/visualizer/validation_five_ui.json) に保存します。
- GitHubのmain `5abecbc7efd8cc626b8bd9d517dd477ad1ae9267` にある元証明書と、使用したローカル証明書のGit blob SHA一致も確認しました。

ChatGPTの旧HTML・PDF・ZIPは、この追加によって変更されません。このフォルダは検証済み証明書を元に作成した完全版です。既存の解析データ・提出資料・証明書への変更はありません。

## 再生成

元証明書は [../../analysis/full_depth4_20261007/](../../analysis/full_depth4_20261007/) にあります。MIA希少フォルダから実行します。

```sh
python3 tools/visualizer/build_visualizer.py
python3 tools/visualizer/build_five_visualizer.py
node tools/visualizer/test_tree_core.cjs --write-report
node tools/visualizer/test_viewer_ui.cjs --write-report
node tools/visualizer/test_five_core.cjs --write-report
node tools/visualizer/test_five_ui.cjs --write-report
python3 tools/visualizer/build_guide.py
python3 tools/visualizer/build_bundle.py
```

HTML・ZIP生成はPython標準ライブラリ、検証はNode.js 18以降を使います。PDFの再生成はPython 3.10以降、reportlab、pypdfが必要です。`build_guide.py --font /path/to/japanese.ttf` で日本語TTFを指定できます。検証の実行に追加のNodeパッケージは不要です。
