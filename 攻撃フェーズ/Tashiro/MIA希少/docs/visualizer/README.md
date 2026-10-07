# MIA希少・適応4回版／非適応5回版の資料

4回版と5回版を、実ID・Team表記で確認するHTML資料です。4回版は第3回の209回答すべてから第4・5・6回の質問と所属配置の確定まで閲覧でき、`analysis/full_depth4_20261007/` の全53軌道証明書を圧縮したまま内蔵しています。

5回版の入口 **[MIA_5query_cards.html](MIA_5query_cards.html)** は、結果で人選を変えない **非適応の固定5回の調査** です。5枚の透明カードと最下段の合成欄が画面内に収まり、カードが1枚ずつ下へ移動して重なります。「所属1人／所属2人」はタブで切り替え、説明用の正解レコードを選べます。

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

1. **所属1人／所属2人** のタブを選びます。5枚は縦に並び、各カードの上に調べる5人のID、この例で含まれる人数、前後の候補数を表示します。主画面をスクロールする必要はありません。
2. **正解レコード（説明用）** でIDを選びます。2人のときは別々のIDを2つ選びます。人数とカードの色は変わりますが、5回の調査対象は変わりません。人数ごとの選択はタブを切り替えても覚えています。
3. **順に重ねる** で、5枚が1枚ずつ最下段へ移動します。**止める／1枚重ねる／1枚戻す／並べ直す** で途中を確認できます。正解やタブを変えるとアニメーションは停止し、5枚を並べ直します。動きを減らす設定が有効な場合は移動を省略します。
4. 各カードの色は、その調査の人数に合う候補。灰色は不一致です。同じ位置に重ねると、1枚でも灰色で覆われる候補は除外され、すべてに合う候補だけが緑で残ります。2人の場合は55組を1組1マスとして並べます。小さな画面では2人組の細かなラベルを省略し、選んだIDと確定IDは主画面に表示します。
5. **詳しく見る** から、数の計算、各人数に合う候補の全列挙、固定コードを開けます。2人組は `11×10÷2＝55`、最初の調査は `15＋30＋10＝55`。初期の2人例は `55→30→15→7→2→1`、1人例は `11→6→3→2→2→1`。詳しい説明だけを別枠の中でスクロールします。
6. 同じ詳細画面の **実際の平均点から6 Teamを照合する** には5行の平均点入力、説明用平均点、入力消去、結果保存があります。この教材の固定セットと提出設定に対応する値を入力します。説明用の正解選択と平均点入力は別々に保持します。

以前の長い資料は **[MIA_5query_reference.html](MIA_5query_reference.html)** に内容を変えず保存しています。計算の元を一度に読む場合に使えます。

資料内の2,494,800は、6 Teamに同じ人を重複させず1・2・1・2・1・2人を割り当てる方法です。`11×45×8×21×5×6＝2,494,800`。216は平均点対応表の回答の種類で、`2³×3³＝216`。教材の11候補・55組とは数える対象が異なります。

今回の固定コードは **[nonadaptive_code.json](nonadaptive_code.json)** に保存しています。過去の資料には固定5bitコードへの言及がありますが、具体列を確認できなかったため、今回新しく構成した非適応案です。旧案の復元とは扱いません。

| 固定調査 | 調べる候補ID |
|---:|---|
| 1 | 76・169・264・462・675 |
| 2 | 47・90・115・264・675 |
| 3 | 76・90・264・383・495 |
| 4 | 47・76・90・169・304 |
| 5 | 47・76・115・462・495 |

11人の5桁の0・1コードがすべて異なり、異なる2人のコードを各桁で整数として足した55個の0・1・2回答列もすべて異なります。2人とも選ばれる回は2と数え、1には丸めません。各調査は5人なので、高側合計に合わせた既知非所属の追加数はTeam順に **1・2・2・6・5・7人** で全5回共通です。

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

非適応5回版の最新サイズと入力ハッシュは [size_report_5query.json](size_report_5query.json)、旧資料は [size_report_5query_reference.json](size_report_5query_reference.json)、従来の適応版は [size_report_5query_adaptive.json](size_report_5query_adaptive.json) に保存しています。すべて単一HTMLで使えます。配布ZIPのファイル名は従来の `MIA_4query_visualizer_bundle.zip` を引き継ぎ、4回版・5回版・旧資料・PDF・説明書・固定コード・検証記録をまとめています。

HTML＋GZ分離ならBase64化の増分を減らせますが、ローカルファイルをそのまま開く際には証明書フォルダの指定や配信の仕組みが必要になります。今回は全53証明書を含めてもHTMLが約1.5 MBなので、**開くだけで使える単一HTML**を選びました。証明書をすべて同時に展開せず、選択した軌道だけを展開し、最大3軌道をキャッシュします。

GitHubは100 MiBを超える通常ファイルを拒否するため、このサイズは通常のGit管理で扱えます。[GitHub公式のファイルサイズ説明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。GZIPの展開には標準の [DecompressionStream](https://compression.spec.whatwg.org/#decompression-stream) を使用します。

## 検証

- 閲覧処理で全209回答・全2,494,800配置を再生し、対称移植、正規化、1,375,470質問ノード、終端の一意性、padding設定を照合しました。最終衝突は0です。[全配置の検証記録](../../tools/visualizer/validation_core.json)
- 生成HTMLの53内蔵GZIPと元証明書のbyte一致、すべてのGZIPの実展開、具体例と別軌道の第6回までの操作、戻る・リセット・平均点入力・JSON/CSV出力・欠落時の停止を確認しました。[操作処理の検証記録](../../tools/visualizer/validation_ui.json)
- HTMLの操作検証はNodeの模擬DOM環境で行いました。ブラウザのローカルファイル表示が実行環境の方針で制限されたため、実ブラウザでの目視確認は未実施です。PDFは全3ページを描画して目視確認しました。
- 5回版は、全11＋55候補・49質問ノード・66葉・深さ上限・padding・216平均点・説明経路を照合しています。[5回版の検証記録](../../tools/visualizer/validation_five_core.json)。操作処理の記録は [validation_five_ui.json](../../tools/visualizer/validation_five_ui.json) に保存しています。
- 従来の適応版2列の教材は、全11人・55組それぞれの5行の質問ID、前後の候補数、回答別の全列挙、候補カードの色と合成結果、途中で確定した行、例の切り替えを照合済みです。[適応教材の検証記録](../../tools/visualizer/validation_lesson_ui.json)
- 非適応版は固定調査、全11人・55組の整数回答列の一意性、330個の途中候補集合、padding、216平均点を照合済みです。[固定コードの検証記録](../../tools/visualizer/validation_fixed_core.json)。全66例×6段階のカード色、タブ・ID選択、アニメーションの順序と中断、異なる画面高での6段の位置、平均点照合1,080ケースの記録は [validation_fixed_ui.json](../../tools/visualizer/validation_fixed_ui.json) に保存しています。旧資料は [validation_fixed_reference_ui.json](../../tools/visualizer/validation_fixed_reference_ui.json) で照合しています。位置は数式と模擬DOMで検証し、実ブラウザの描画確認とは区別しています。
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
node tools/visualizer/test_fixed_reference_ui.cjs --write-report
python3 tools/visualizer/build_guide.py
python3 tools/visualizer/build_bundle.py
```

HTML・ZIP生成はPython標準ライブラリ、検証はNode.js 18以降を使います。PDFの再生成はPython 3.10以降、reportlab、pypdfが必要です。`build_guide.py --font /path/to/japanese.ttf` で日本語TTFを指定できます。検証の実行に追加のNodeパッケージは不要です。
