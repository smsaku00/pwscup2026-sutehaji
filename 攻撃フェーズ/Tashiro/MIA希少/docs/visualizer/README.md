# MIA希少・第3〜第6回の完全決定木ビューア

第3回の209回答すべてから、第4・5・6回の質問と所属配置の確定まで閲覧できます。`analysis/full_depth4_20261007/` の全53軌道証明書を圧縮したまま内蔵した、単一HTMLです。

## 開き方

1. [MIA_4query_tree.html](MIA_4query_tree.html) をGitHubの **Download raw file** で保存し、最新版のChrome・Edge・Safariで開きます。GitHubのファイル画面ではHTMLのソースが表示されます。
2. **具体例を1手進める** で、`2,494,800 → 40,428 → 631 → 20 → 1` を確認できます。**最後の20候補を見る** で第6回の直前へ移動できます。
3. 任意の回答行を選ぶと、その枝の次の質問が開きます。**1手戻る**、**最初に戻る**、**犬の例／実ID・Team** の切り替えもできます。
4. **平均点から回答を選ぶ** は、保存済みの216回答の点数表で復号した後、現在の枝にその回答が存在するかを検査します。
5. **現在の経路を保存** は回答codeと結果をJSONに、**残る全候補をCSVで保存** は現在の候補配置と予想回答をCSVに書き出します。

HTMLを開けば全データを利用でき、サーバー起動や証明書フォルダの選択は不要です。外部ライブラリの読み込みもありません。PDFのリンクも使う場合は、同じフォルダに [MIA_4query_tree_guide.pdf](MIA_4query_tree_guide.pdf) を保存してください。[配布ZIP](MIA_4query_visualizer_bundle.zip) を展開すると、HTML・PDF・説明書・サイズ記録・検証記録が揃います。

具体例は説明用の仮定経路で、実際の所属結果ではありません。犬1〜11は候補ID順、飼い主A〜FはTeam 6・10・11・16・17・25に対応します。所属人数は1・2・1・2・1・2で、11候補のうち2候補はどのチームにも属しません。

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
| 完成した単一HTML全体 | 1,501,180 | 1.501 |
| 単一HTMLをGZIP圧縮した場合（参考実測） | 1,108,583 | 1.109 |
| 3ページのガイドPDF | 115,156 | 0.115 |

HTML＋GZ分離ならBase64化の増分を減らせますが、ローカルファイルをそのまま開く際には証明書フォルダの指定や配信の仕組みが必要になります。今回は全53証明書を含めてもHTMLが約1.5 MBなので、**開くだけで使える単一HTML**を選びました。証明書をすべて同時に展開せず、選択した軌道だけを展開し、最大3軌道をキャッシュします。

GitHubは100 MiBを超える通常ファイルを拒否するため、このサイズは通常のGit管理で扱えます。[GitHub公式のファイルサイズ説明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。GZIPの展開には標準の [DecompressionStream](https://compression.spec.whatwg.org/#decompression-stream) を使用します。

## 検証

- 閲覧処理で全209回答・全2,494,800配置を再生し、対称移植、正規化、1,375,470質問ノード、終端の一意性、padding設定を照合しました。最終衝突は0です。[全配置の検証記録](../../tools/visualizer/validation_core.json)
- 生成HTMLの53内蔵GZIPと元証明書のbyte一致、すべてのGZIPの実展開、具体例と別軌道の第6回までの操作、戻る・リセット・平均点入力・JSON/CSV出力・欠落時の停止を確認しました。[操作処理の検証記録](../../tools/visualizer/validation_ui.json)
- HTMLの操作検証はNodeの模擬DOM環境で行いました。ブラウザのローカルファイル表示が実行環境の方針で制限されたため、実ブラウザでの目視確認は未実施です。PDFは全3ページを描画して目視確認しました。
- GitHubのmain `5abecbc7efd8cc626b8bd9d517dd477ad1ae9267` にある元証明書と、使用したローカル証明書のGit blob SHA一致も確認しました。

ChatGPTの旧HTML・PDF・ZIPは、この追加によって変更されません。このフォルダは検証済み証明書を元に作成した完全版です。既存の解析データ・提出資料・証明書への変更はありません。

## 再生成

元証明書は [../../analysis/full_depth4_20261007/](../../analysis/full_depth4_20261007/) にあります。MIA希少フォルダから実行します。

```sh
python3 tools/visualizer/build_visualizer.py
node tools/visualizer/test_tree_core.cjs --write-report
node tools/visualizer/test_viewer_ui.cjs --write-report
python3 tools/visualizer/build_guide.py
python3 tools/visualizer/build_bundle.py
```

HTML・ZIP生成はPython標準ライブラリ、検証はNode.js 18以降を使います。PDFの再生成はPython 3.10以降、reportlab、pypdfが必要です。`build_guide.py --font /path/to/japanese.ttf` で日本語TTFを指定できます。検証の実行に追加のNodeパッケージは不要です。
