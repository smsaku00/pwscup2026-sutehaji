# 2026-10-07以前の保存物・旧実装の調査記録

最終更新: 2026-10-08。既存の保存物・アーカイブは変更していない。過去の集計、保存予測、公開入力だけで動く既存純関数を読み、予備戦B_selfの1049人・自チーム以外の20対象で公式TPRを確認した。最終一覧からのTeam12除外は親集計で行う。この旧方式補充は現在のMIA全体実験本体とは別の収集器である。

現在の予備戦A_bgは5143行。READMEの旧10286行という記載は使わない。全再評価はルート `.venv/bin/python -B` を使い、旧main・旧Python固定パス・旧harness・加工物の生成器を実行していない。所属正解は採点時にだけ渡した。C対A分類器の学習ラベルと、B候補の所属正解は区別した。

## 読み取り可能な旧保存物と今回の扱い

| 保存元（プロジェクト相対） | 構造・内容 | 今回の扱い |
|---|---|---|
| `予備戦/結果分析/集計/過去8回_対象別.csv` | 8回×20対象、`method,target,mia_all,n_pos` 等 | 保存F_miaを現公式で再採点し160個の保存TPRと整合。全体不変のAIA/希少変更を同一方式の別名として第1/4/5/6回、第2/3回、第7/8回に集約 |
| `予備戦/攻撃フェーズ/提出記録/*/F_mia.csv` | record_idと対象confidence | B_selfのrecord_id順へ整列して採点。提出や回答更新は行わない |
| `予備戦/結果分析/集計/未採用手法_対象別.csv` | 44条件×20対象、全体/希少TPR | 全体生順位は以下のJSONから採点。Gen7/9/10非希少差替3条件は保存TPR参照、公開真値確認後のTeam1/2固定を含む評価A/oracle参考。AIA・希少だけの変更は独立した全体方式に数えない |
| `予備戦/攻撃フェーズ/分析/21_MIA全体と希少のGenLRA比較/比較結果.json` | `record_ids`, `raw_scores[team][round6/qi7_gen/qi7_distance/qi7_point/attrs10_gen/attrs10_distance/attrs10_point]` | round6の別名を除き全6生scoreを再採点 |
| `予備戦/攻撃フェーズ/分析/22_GenLRAの7属性9属性10属性比較/比較結果.json` | `raw_new_scores[team][attrs9/smoking_kernel_only/smoking_neighbors_only]` | 3生scoreを再採点 |
| `本戦/攻撃フェーズ/Tashiro/20261005_MIA初回_GenLRA比較/結果/予備戦_対象別評価.csv` | 8構成×20対象、`variant,team,members,mia_all` | `gen_lra.calculate`の純関数を再計算、160件の保存TPRと全一致 |
| `アーカイブ/予備戦_攻撃フェーズ/過去のコード/mia_domias/{out,out_prior,out_w07}/F_mia.csv` | 20対象confidenceと`domias.json` | 保存回答を再採点。帯域Scott×1、専用非会員0、県事前重み0/1/.7。初期outのprior_weight欠落は0 |
| `アーカイブ/予備戦_攻撃フェーズ/過去のコード/mia_domias/zhou/F_mia_zhou.csv` | 独立周辺KDE・カテゴリ密度比の保存回答 | 連続7列、カテゴリ6列、C1公開QI近接120行の除外を含む保存回答を再採点 |
| `アーカイブ/予備戦_攻撃フェーズ/work_20260901以前/prelim_attack/{r1,r2}/F_mia.csv`, `prelim_attack2/r1/F_mia.csv` | 旧Cross-C・公開構造・指紋組込の保存回答 | 現所属真値で再採点。旧proxyやtime指紋の説明が正しいことの証明には使わない |
| `アーカイブ/予備戦_攻撃フェーズ/work_20260901以前/F_mia.csv` | 練習課題の30列・29対象 | 予備戦20対象の列と一致しないため除外。record_idが同じに見えても転用しない |

第2/3回の生成属性条件は保存物から確定できない。その欄は未確定と表示し、保存confidenceの公式採点が確認できたことと分けた。

## 公開入力で再実行した既存純関数

`historical_results.py` は86条件×20対象=1720行を返す。保存予測からの採点に加えて次を含む。

- `work_20260901以前/attacks.py`: DOMIAS、clinical6値一致、NN gap（kC1/kA2）、kNN15、県周辺事前、既定順位ensembleの6方式。
- `work_20260901以前/cross_c_mia/cross_c_consensus_mia.py`: raw近さ、公開帰無z、cross specificity、rank residual、k5 specificity、自チームC5経由bridgeの6方式。C近接A除外閾値.05、整合対照1049人、元seed20260829を維持。全20対象Cを参照するtransductive条件。
- `過去のコード/densratio/densities.py`: `ratio_zoo()`の13設定。KDE4帯域、単一Gauss、GMM BIC、kNN25/5、二次logistic5fold、HGB5fold、uLSIF、RuLSIF、Gaussian copula。所属ラベルによる学習・選択を行わない。GMM/BIC、uLSIFの内部選択は元方式の定義のまま。
- `work_20260901以前/mia_redteam.py`: 既定shadow分類器と臨床5viewのtail linkage。`outcome_mia_redteam.py`: event/time一致・近接、joint NN・kNN、生存尤度、Cox residual、元3split×3generatorの既定9seed分類器とその各ensemble。旧proxyラベルに依存したjitterは加えず、公式の同点を保持。
- `work_20260901以前/submit_audit.py`: timeを含む連続8列のNN gap。
- `work_20260901以前/aia2mia/bench.py`: 同性内QI5のd1/r12/dr1/dr5。proxy対照の得点を流用せず、同じ数値score定義をB候補へ適用。

`historical_virtual.py` は10条件×20対象=200行を返す。`virtual_mia_engine.py`の既定7profileのうちclassic_generalistは旧library ensembleと成分・重み・gateが同じ明示的別名のため集約。残6固定profileと、既定seedの臨床/joint shadow2成分、公開source回復のgap/selected-negative2成分を評価した。source回復は公開A/Cの分位写像、年齢40–74、最近傍k100から推定する既存手順で、私的な真のsource indexを与えていない。

`historical_atk.py` は15条件・249行を返す。公開入力だけで再現できる12条件は全20対象、元seed1半分割3条件は元の24/26/29だけである。

- `atk19222023/mia2.py`: kA1距離比eps1e-6、k5距離比、log KDE floor1e-300、raw NN8。
- `atk19/mia.py`: QI5のsex/県block NN。欠落blockを最大有限距離へ置換する元定義を維持。
- `atk242629/p6_mia.py`: 対象C近接.05除去後のAで標準化し直すNN gap、kNN15/3、DOMIAS。
- `atk242629b/mia.py`: 元A尺度のまま除染Aを背景にするNN gap、kNN15、sex/県block NN（欠落時同性へfallback）。DOMIASは上記の同一純関数/入力の明示的別名。元の非東京都だけの診断から、score純関数自体を変えずB全1049へ適用。
- `atk242629b/mia2.py`: 元rng種1を24→26→29の順に進め、permutation・整合対照抽出のrng消費も再現したA_ref半分割のNN gap/kNN15/DOMIAS。残17対象に新しいrng順を設けない。

`atk1213/p2_mia.py`、`p3_mia2.py`、`atk2346/mia.py`、`atk19222023/mia.py`等のraw NNのproxy tail、対照抽出を変えたAUC、固定距離閾値の件数診断は、raw NN7を新方式に数えない。真の所属TPRが保存されていないことを、TPR=0として補完することもしていない。

同順位になっただけの異なるアルゴリズムは統合していない。例えば公開帰無zとraw NN、密度zooのKDEとlibrary DOMIASは、元の推定・変換条件が異なるため各recordを残す。明示的なコピー、同一純関数/入力、全体不変のAIA/希少変更だけが別名集約の対象。

## 実行しない、または未定義の条件

| 保存元/条件 | 理由 |
|---|---|
| `cross_c_mia` のcapacity26 | 26人を強制充填する割当集合。元定義の連続confidenceがないため、新スコアへ作り替えない |
| `aia2mia/bench.py` のd1_pref | 各AIA challengeの地元県へ固定する診断。B全体への県条件を変えると元の定義を離れる |
| `atk242629b/mia2.py` 半分割の残17対象 | 元rng消費順は24/26/29だけ。未実行対象へ新たな順・seedを足さない |
| `atk19222023/mia2.py` のbinmatch | 量子化gridの一致件数が診断出力。元の個別membership confidence保存物がない |
| `whitebox_audit.py` targeted LOO | 対象の私的Bの他の全行と同クラスreplacementを使う加工再生成oracle。公開C/Aだけでは実行できない。公開source回復2成分は別途評価済み |
| 旧 `densratio/mia_results.csv`, PanelA2/PanelB2、aia2mia、atk、仮想大会の保存proxy TPR/AUC | 自己C・代替対照・別world・擬似陽性・仮想加工の評価。大会20対象の所属TPRとして転用不可。入力だけで定義できるscoreは上記収集器で補充 |
| 本戦バトルの別チームB/別加工worldの得点 | 対象・所属真値が予備戦20対象と違う。別worldの得点を大会チームへ割り当てない |

未知条件・入力不足を未達や攻撃不可能という結論へ変換しない。上記は定義と入力の限界であり、未実行の探索による改良余地の否定ではない。旧アーカイブの全AIA診断・防御生成器・疑似世界を、新規MIA攻撃として再定義する作業は行っていない。

## 最終cacheの確認

3つのcacheは方式×対象の集計値と条件情報だけを保存し、個人別scoreや所属ラベルを含まない。使用したB、A、対象C、pool_ids正解、旧コード、現在の公式scoringをSHA-256で照合する。`gather(root)`のcache読取でハッシュ一致と必要metadata全keyの存在を確認済み。

旧方式の帯2以上（TPR生値>0.25）はTeam1/2だけ。Team1/2/12以外の最大は `legacy_redteam_event_time_exact` のTeam4、TPR=0.16513882352941175。virtual補充の最大はTeam17=.08、atk補充の最大はTeam26=.12。これは上記で確認できた条件についての結果で、未定義条件までの否定ではない。
