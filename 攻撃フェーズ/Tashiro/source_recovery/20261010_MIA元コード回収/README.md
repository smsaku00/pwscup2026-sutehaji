# MIA元コード回収（2026-10-10）

## 承認された過去入力2件の追記

再現に必要な第6回MIA回答CSVと第7回計算条件JSONの2件だけを、本人の承認範囲として原本のまま追加しました。JSONの固定レコードID別履歴も含めて承認された入力です。配置は `reproduction_inputs/` 以下に元の相対パスを保持しています。これまで回収しなかった過去回答CSV・履歴付きJSONの掲載について、この2件だけが今回の例外です。

| 元入力 | 掲載ファイル | bytes | 原本SHA256 |
|---|---|---:|---|
| 第6回MIA回答CSV | [F_mia.csv](reproduction_inputs/予備戦/攻撃フェーズ/提出記録/6_20260907_GenLRA_Zhou/F_mia.csv) | 350,924 | `bb71413e27ff0b005c21dfcd52a24c5a30952b1eb552ae5cdeb585cbd66644e8` |
| 第7回計算条件JSON | [計算条件.json](reproduction_inputs/予備戦/攻撃フェーズ/提出記録/7_20260907_GenLRA_Seo/MIA/計算条件.json) | 33,920 | `1d7f4d369fecf17f2a059bcbe378ea6b061e9fb29cad250e4a0d94c45393be9b` |

回収元は `/Users/taisei1105/pws2026-local` の表中ファイルの元相対パスです。CSVは1,049行・21列（record_idと20対象）で、列数とrecord_idの一意性を確認しました。JSONは指定サイズ・指定SHA256に一致し、その `baseline` は上記第6回CSVの元相対パスと一致します。CSVの期待ハッシュは指定されていないため、表の値は今回読み取った原本ハッシュです。

秘密情報に関連するJSONキー、秘密鍵マーカー、認証トークンの既知形式、パスワード等の代入を点検し、該当は0件でした。データ・履歴は加工していません。個人の実データを追加収集していません。秘密情報検査は形式検査のため、すべての未知形式の不存在を保証するものではありません。

[manifest.json](manifest.json) と [dot_manifest.json](dot_manifest.json) の `approved_historical_inputs.files` に原本パス・掲載パス・サイズ・ハッシュがあります。`source_files=260` は前回までのコード・設定原本の件数のままで、今回の2入力は `approved_historical_input_files=2` として別に記録しています。実行用の別ツリーには、この2入力を `original_relative_path` の位置に無加工で配置してください。掲載原本260件や現在の提出CSVを上書きしないでください。

今回はコピー・CSV/JSON形式確認・ハッシュ照合・秘密情報検査だけです。新しい攻撃計算、採点、現在の本戦提出の作成・差し替えは行っていません。以下の「初回回収」「9件追跡」はそれぞれ当時の記録です。


## 9件の生成元追跡の追記

初回の252原本を保持したまま、8原本（Python7、MJS1）を追加し、現在は260原本です。[provenance_followup_9/index.json](provenance_followup_9/index.json) が今回の入口です。[targets.json](provenance_followup_9/targets.json) に各保存スコアのsource ID、元コードとハッシュ、固定条件、入力・出力のハッシュ対応、確認根拠と未確認部分を記載しました。

| 対象 | 生成元の確認 | 根拠・制約 |
|---|---|---|
| 古い `prelim_attack/r1`（S0058） | `work_20260901以前/build_attack.py` を回収 | docstringの出力先と保存値の希少配置がround=1 mod10で対応。保存CSVは0セルなし。原コマンドのログ・当時の入力ハッシュは未確認 |
| 古い `prelim_attack2/r1`（S0060） | `work_20260901以前/build_attack2.py` を回収 | 保存reportのround=1、出力JSONのキー、非会員163行の0化、空状態での指名式と保存指名の一致。実行前のSTATE原本は未回収 |
| `F_mia_zhou`（S0088） | コード本体は未特定 | README記載の出力SHA256と保存CSVは一致。記載コード `mia_mixed_density_attack_improved.py`、除外候補CSVはPWSツリー内に見つからず |
| 提出1（S0098） | コード本体は未特定 | `mia.md` と手法メモで設定の一部を確認。説明から推定実装は作らない |
| 提出2（S0099） | コード・固定設定は未特定 | 保存CSVとZIPのみ。ZIPにコードは含まれない |
| 提出7（S0100） | 作成時記録の `MIA/作成.py`、検証Python、CSV作成MJSを回収 | 3コードのSHA256は作成時の入力記録と移管記録に一致。作成時ログの出力SHA256も提出7保存CSVに一致 |
| Gen7差替（S0102） | `予備戦/結果分析/分析.py` の合成処理を回収 | analysis21 `qi7_gen` を使用 |
| Gen9差替（S0103） | 同上 | analysis22 `attrs9` を使用 |
| Gen10差替（S0101） | 同上 | analysis21 `attrs10_gen` を使用 |

生成コードまたは合成処理を特定したのは6/9件です。これは今回の数値再現を意味しません。未特定3件と、追加の未回収入力は [search_and_missing.json](provenance_followup_9/search_and_missing.json) に分けて記載しています。検索は同じPWS作業ツリーのコードと保存記録を対象にし、PWS外のDownloads・メール添付やメンバー個人の作業環境は調べていません。

前回の `予備選分析_スクリプト/build_attack2.py` は、S0060の生成元と同名でも異なる内容です。S0060には今回追加した `work_20260901以前/build_attack2.py` を使います。一方、`build_attack.py` のこの2保存場所は同じハッシュでした。原本の `main()` は回答CSVやtokenも書くため、今回実行していません。

旧データ配置 `work/data_prelim_team05/` と `share_team05/data/` は現存しません。同名入力は `予備戦/加工フェーズ/分析/資料/加工コード_share_team05/data/` にあり、現在の予備戦公式B_self/A_bgとハッシュが同じです。ただし、S0058/60生成時の入力ハッシュ記録がないため、当時の入力と完全一致したと断定しません。パスの対応候補と現在の入力ハッシュを [input_hashes.json](provenance_followup_9/input_hashes.json) に記録しました。

提出7の保管場所は `予備戦/攻撃フェーズ/提出記録/7_20260907_GenLRA_Seo/作成時記録/MIA/`、元の実行場所は `予備戦/攻撃フェーズ/提出準備/7_20260907_GenLRA7_9_MIA/` です。移管記録で3コードのハッシュを照合しました。別の実行用ツリーに元の相対配置を復元してから使用してください。保管深さのまま起動すると `HERE.parents[3]` が誤ったROOTを指します。[dependency_graph.json](provenance_followup_9/dependency_graph.json) の `source_location_aliases` に対応を載せています。原本は変更していません。CSV生成は同梱Nodeと `@oai/artifact-tool` に依存します。現在観測した版2.8.89は当時の版の証明ではありません。計算条件JSONは個人単位の履歴を含むため、原本ハッシュと計算設定だけの [projection](provenance_followup_9/submission7_fixed_settings.json) を収録しました。

Gen7/9/10の3合成は、公式採用 `cohort05_F.zip!F_mia.csv` を基準にしています。このメンバー内容が保管された第8回F_miaと全バイト一致することを確認しました。Team1/2と希少を維持し、残18対象の非希少・非専用875行に `.1+.8*rankdata(raw[mask], method='average')/876` を代入し、非希少の専用行を0にします。この式は提出7のcap依存・min順位/875の変換と異なります。analysis21/22の保存生スコアが参照する配布入力の各24ハッシュは現ファイルと一致しました。元のコード・固定設定・保存入力との対応だけを確認し、合成スコアの計算や採点は行っていません。

以下の初回回収の説明で「252」「238」は初回時点の数です。現在の全件数は260（Python245、計算JSON11、requirements3、MJS1）で、最新の一覧は `manifest.json` にあります。初回の252原本・カタログ93片は変更していません。

## 初回回収の記録

このフォルダは、Mac上の原本コード・計算設定・依存コードを外部環境に渡すための回収物です。今回、MIA実験、方式探索、採点、提出物作成、依存導入は行っていません。以前の背景1・2・3比較の未掲載実験や、他の未掲載レポートは含めていません。

掲載先は `smsaku00/pwscup2026-sutehaji` の専用ブランチ `codex/mia-source-recovery-20261010`、追加範囲は `攻撃フェーズ/Tashiro/source_recovery/20261010_MIA元コード回収/` だけです。原本と既存リポジトリファイルは保持しています。

## dot側の入口

最初に [dot_manifest.json](dot_manifest.json) を取得してください。指定24ファイルの所在・ハッシュ、全252原本の [manifest.json](manifest.json)、依存グラフの小分けファイル、カタログの索引、検証記録への参照を載せています。GitHubの読取はブランチまたは最終応答のcommit SHAを明示してください。mainには今回の追加がありません。

- `originals/`：原本252ファイル（Python238、計算設定JSON11、requirements3）。元ルートからの相対階層を保持し、内容をバイト単位でコピー。
- `manifest.json`：元相対パス、掲載相対パス、SHA256、サイズ、分類。
- `dependency_graph_parts/`：静的import・Pythonパス文字列・CSV/NPZ/JSON等の入出力候補。文字列には出力ファイルも含まれます。
- `catalog/catalog_index.json`：既存14,864条件の原本ハッシュと各分割片の順序・ハッシュ・ヘッダー仕様。
- `environment.json`：回収時点のPython・直接ライブラリの観測版。派生の `requirements-reproduction.txt` には計算用の固定版を記載。
- `missing_files.json`：指定原本の欠落は0。意図してコピーしなかった実行入力を別記。
- `verification.json`：静的・原本照合・カタログ復元の検証結果。

## 回収元と範囲

元ルートは `/Users/taisei1105/pws2026-local`。主な原本は次の2フォルダです。

```
本戦/攻撃フェーズ/Tashiro/20261008_MIA全体_朝9時までの方式探索/
本戦/攻撃フェーズ/Tashiro/20261008_MIA全体_較正の広範比較/
```

両フォルダ内のPythonコード（技術引継ぎ・失敗時を含む保存スナップショットも区別して収録）と計算方針JSONを回収しました。共通・較正コードが参照する10月7日の分析、Gen-LRA、指定された旧攻撃コードのローカルimport依存も収録しました。旧密度比の呼出側、過去Gen-LRAの参照コード、参照される3時期の公式Pythonキットも原本のまま収録しています。公式schemaのコードはコピーだけで、schemaやμΣの変更はしていません。

Pythonキットの版はフォルダ別に保っています。`スターターキット/初版/`、`スターターキット/改訂版_20260824/`、`本戦/配布データ/pwscup2026-kit-main_revised0912/` を混同しないでください。原本の関数が指定する版を使います。

データCSV、会員正解、特徴・順位NPZ、計算結果のJSON/レポート、回答CSV、提出ZIP、token.txt、.env、鍵、仮想環境、キャッシュは回収していません。CSVとして収録したものは明示指定された方式カタログと出典メタデータだけです。原本コード内の `token.txt` 等への参照は原本の一部として残しますが、秘密値は含めていません。

## カタログをテキストで読む・原本に戻す

原本：`攻撃フェーズ/Tashiro/MIA全体/experiments/20261008_帯別横断検証/方式条件一覧.csv`。18,450,183バイト、14,864データ行、SHA256：

```
cee31ab588cd943aa8b34d560d02dd249d8ab7d5cba37c02dcb5d07a89185cb3
```

`catalog/text_parts/` の93片は各200,000バイト未満（最大199,974バイト）です。UTF-8の文字・CSVレコードの途中で分割せず、元のBOM、ヘッダー、改行を保持します。各片には同じヘッダーが1つあります。`catalog_index.json` の `text_parts` 配列順で取得し、最初のヘッダーだけを残して、2片目以降の先頭 `header_bytes` バイトを除き、残りをそのまま連結します。CSVをパースして再出力すると引用符や改行が変わり得るため、復元では再シリアライズしません。

以下は元コードを動かさない復元・照合コマンドです。既存出力先は拒否します。

```sh
/path/to/.venv/bin/python -B tools/restore_catalog.py --mode text --output /tmp/mia-method-catalog.csv
/path/to/.venv/bin/python -B tools/verify_recovery.py
```

gzip全体版（435,451バイト）とgzip30分割版もあります。`--mode gzip`、`--mode gzip-parts` で同じ原本に戻せます。3方式の復元結果は全バイト一致し、原本SHA256とも一致しました。`catalog/結果保存元.csv` は既存336出典のパス・ハッシュメタデータです。その出典ファイル本体や保存得点を今回新たに掲載したものではありません。カタログ内の評価欄は過去の探索記録であり、今回の結果ではありません。

## 別環境での準備

1. 掲載commitのフォルダを取得し、上記 `verify_recovery.py` で原本ハッシュ・構文・復元を照合します。Mac原本にアクセスできる場合は `--source-root /Users/taisei1105/pws2026-local` で原本との二重照合もできます。
2. 外部の作業環境にPython 3.12の専用 `.venv` を準備します。計算用依存は同じPythonの `-m pip` で `requirements-reproduction.txt` を使い、原本のrequirementsも確認してください。今回Macで依存の新規導入・仮想環境作成はしていません。
3. privateリポジトリ内の既存 `公式データ/` を、変更せずに使える状態にします。古いスターターのparticipant_dataや過去の保存結果を必要とする方式には、その正しい元入力も別途必要です。
4. 絶対パス対応が必要なら、以下の準備専用ラッパーを使用します。掲載原本を編集せず、別の新しいディレクトリに実行用コピーを作り、そのコピー内だけ元ルートの絶対パスを変更します。出力先はリポジトリの外にしてください。

```sh
/path/to/.venv/bin/python -B tools/prepare_runtime.py \
  --output /tmp/mia-source-runtime \
  --repository /path/to/pwscup2026-sutehaji \
  --python /path/to/.venv/bin/python
```

ラッパーは原本252ファイルを照合してからコピーし、`.venv`、旧`work`、予備戦配布データ、本戦入力の対応を作ります。公式データへのリンクは実行用コピー側だけです。本戦のtokenはリンクしません。変更されたコピーのハッシュとパス対応は `/tmp/mia-source-runtime/runtime_preparation.json` に記録します。準備後の実行には `/tmp/mia-source-runtime/.venv/bin/python -B` を使い、root `.venv` を確認する原本の条件を満たします。ラッパー自身は原本をimport・実行しません。

Mac上でこの準備ラッパーのコピー・パス対応・既存出力拒否を確認しました。MIA計算は起動していません。古い計算結果の再集計や本戦の既存回答への照合を行うentrypointは、未収録のCSV/NPZ/JSONを要求するため、この回収物だけではそのまま実行できません。元の計算設定と対応する入力が揃うまで実行せず、必要な入力は `dependency_graph_parts/` とコード本体で確認します。

## 固定3方式を確認する際の入口

`q7_sex_bins40_s10_logratio__county500__cross0` は探索フォルダの `階層ヒストグラムの探索.py::histograms` です。q7の数値5列と性別・県、性別条件、bins=40、s=10のlogratioを取り、`共通.py::variants` のcounty500、`matrices` のcross=0を使用します。背景群への固定縮約50も原本の定義の一部です。

`q13_quantile_sex1_kde_ratio_h0.5__raw__cross0` は `分布を揃えた距離の探索.py::align` のquantile/by_sex=Trueと `距離と分布の探索.py::geometry` にあります。q13の属性定義、TG/ALTのlog処理、IQR尺度、元の自動帯域×0.5、カテゴリ混合、raw順位、cross=0を原本どおり確認します。`混合分布と局所密度の探索.py` も依存する共通幾何関数は同じファイルに収録しています。

P01は10月7日 `分析.py::core` のGen-LRA7（k=10、rho=.98）と10月8日較正 `分析.py` の `flat_pref_sex_s500`、他対象を除いた中央値係数0.1を確認します。full背景は全5,143行です。較正用の他対象公開Qと密度用BGは別の入力です。

これらの固定方式の数値再現、背景1・2・3の比較、予備戦保存full値との照合、本戦順位の変化は外部での次工程です。今回は一致すると推測したり、以前の結果を今回の検証結果として扱ったりしていません。元探索コードの `main()` は多数方式を一括実行するため、固定方式の再現では必要な関数と保存設定を選んで使用してください。本戦のTPRは正解非公開のため報告できません。

## 検証の範囲と制約

原本252ファイルとのバイト比較、Python238ファイルの静的構文コンパイル、設定JSON11ファイルの読取、3通りのカタログ復元、秘密値パターンと禁止ファイルの点検を行いました。元コードのimport、MIA実行、乱数依存の数値一致、全14,864条件の機能再現は行っていません。静的依存追跡と秘密値パターン検査には検出範囲の限界があります。

指定24原本の欠落は0です。保存結果を入力として使う集計系、過去の回答照合、旧仮想攻撃ハーネス、失敗時のスナップショットには追加入力や実行順の条件があります。全カタログの設定を保管できたことと、全条件が入力なしで実行できることは別です。フォルダ別のコード版と元設定を固定し、次工程でも新しいパラメータ探索と保存値の再現を区別してください。
