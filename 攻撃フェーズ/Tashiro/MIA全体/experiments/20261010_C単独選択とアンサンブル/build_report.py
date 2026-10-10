from pathlib import Path
import json,hashlib,zipfile
import pandas as pd
R=Path(__file__).resolve().parent
s=pd.read_csv(R/'summary.csv');b=pd.read_csv(R/'heldout_by_target.csv');f=pd.read_csv(R/'C_only_features.csv')
labels={
'global_train_best':'訓練対象で選ぶ単一方式（基準）','C_gate_fixed_k3_weight05':'Cのみ・3近傍/局所50%固定','C_gate_nested':'Cのみ・内側LOTOで近傍数/局所比率選択','oracle_infeasible':'対象正解で後出し最良（実行不能oracle）',
'ensemble_train_champion_mean':'訓練側8家族代表・順位平均','ensemble_train_champion_median':'訓練側8家族代表・順位中央値','ensemble_train_champion_max':'訓練側8家族代表・順位最大値','ensemble_train_top3_families_mean':'訓練側上位3家族代表・順位平均','ensemble_train_top5_families_mean':'訓練側上位5家族代表・順位平均','ensemble_train_champion_weighted':'訓練成績で重み付けする8家族代表',
'fixed_uniform_all_candidates':'固定・全43,148候補順位平均','fixed_equal_family_background_mean':'固定・家族/BG等重み順位平均','fixed_family_consensus_median':'固定・8家族平均の中央値','fixed_family_consensus_max':'固定・8家族平均の最大値'}
def table(view):
    h=s[s.view==view];out=['| 方式 | 平均TPR | ≥0.10 | >0.10 | 基準との差 | 勝/分/敗 |','|---|---:|---:|---:|---:|---:|']
    for _,x in h.iterrows():out.append(f"| {labels[x.strategy]} | {x.mean_tpr:.6f} | {x.TPR_ge010}/{x.targets} | {x.TPR_gt010}/{x.targets} | {x.mean_delta_train_global:+.6f} | {x.wins}/{x.ties}/{x.losses} |")
    return '\n'.join(out)
wide=b[b.scope=='train_and_test_without1_2'].pivot(index='team',columns='strategy',values='tpr_0.01')
chosen=['global_train_best','C_gate_fixed_k3_weight05','C_gate_nested','ensemble_train_champion_mean','fixed_equal_family_background_mean','oracle_infeasible']
wide[chosen].to_csv(R/'primary18_team_comparison.csv',float_format='%.17g')
pt=['| Team | 単一方式基準 | 固定Cゲート | 内側調整Cゲート | 8家族代表平均 | 固定家族/BG平均 | oracle |','|---|---:|---:|---:|---:|---:|---:|']
for team,x in wide.iterrows():pt.append('| '+str(team)+' | '+' | '.join(f'{x[n]:.6f}' for n in chosen)+' |')
text='''# 加工データCだけによるMIA方式選択とアンサンブル

## 結論（Team 1・2を除いた18対象を重視）

- Cだけでも、粗い離散化、重複、相関の弱化/強化などの加工上の違いは観測できる。ただし、今回の19個の構造特徴から「効く攻撃方式」を十分に選べることは確認できなかった。真の加工アルゴリズム名を一意に当てる検証ではない。
- Team 1・2を訓練にも評価にも入れない対象LOTOでは、単一方式基準0.004444に対し、C-only固定ゲート0.013162、内側LOTO調整ゲート0.012919。平均値は小幅に改善したが、TPR 0.10超はどちらも0/18対象。
- アンサンブルは限定的に改善した。各訓練foldで8家族から代表を1個ずつ選び、順位を等重み平均すると0.021801、基準差+0.017357、7勝10分1敗。Team 13が0.12となり、0.10超は1/18対象。ただし中央値は0のままで、多くの対象を救うほどの効果はない。
- 全候補の無差別平均は0.012912、家族/BG等重み平均は0.015135。今回の小さい探索での数値比較であり、最良に見えたアンサンブルを未使用データで確認した結果ではない。
- 同じ43,148候補から評価対象の正解を見て毎回最良を選ぶoracleは平均0.147866、18/18対象で0.10超。この差は「良い候補が存在する」と「Cから選べる」の違いを示す。oracleは配備不可。

## 1. 比較条件

- 評価は予備戦。BGの `main` は本戦由来の背景データを意味するだけで、本戦の正解を評価したものではない。
- 推論時の基礎予測列は全20対象を保ったまま固定。主評価は1/2以外の18対象について、外側1対象を完全に保留し、残る17対象だけで学習/選択する。
- C-only調整版の内側検証は訓練17対象のLOTOであり、外側対象は内側の訓練にも検証にも入らない。
- 指標は公式互換の線形補間TPR@FPR=1%。全1049人で採点。875人のnormal候補で順位を揃え、private=0、private以外のrare=0.4、normal=0.6+0.3×min-rank/875の既存規則を適用する。
- `normal875` / `nonrare1025` / `all_fixed1049` を混ぜていない。追加の全20対象LOTOと、その予測から1/2だけ集計除外した18対象表示は、主評価と明確に分ける。
- 閾値計数は浮動小数点誤差1e-12を許容し、≥0.10と>0.10を別列に残した。

### 基礎候補集合

公開入力だけから作られ、全7BGで完成済みの8家族を全設定採用した。今回の全対象TPRで候補をtop Kに絞っていない。

| 家族 | 1BG当たり設定数 |
|---|---:|
| histogram | 1920 |
| geometry | 666 |
| alignment | 750 |
| mixture | 960 |
| local | 300 |
| direct | 1008 |
| attribute_seeds | 320 |
| stable_centers | 240 |

合計6164設定×7BG=43,148候補。BGはstarter、prelim、main、starter+prelim、starter+main、prelim+main、starter+prelim+main。

学習済みmembershipモデル、policy、過去の対象別採用履歴を使う合成は除外した。元のcross補正は全対象の公開Cを利用するtransductiveな処理であり、公開情報を使うこと自体は保留違反として扱っていない。

## 2. Cしか見ない方式選択

### C-only特徴（19個）

Cそのものから、欠損率、全属性/QI7重複率、数値の一意値率（平均/最小）、0.1/1/5/10刻み割合、カテゴリエントロピー/最大頻度/種類数、数値Spearman絶対相関（平均/最大）、カテゴリ間の正規化相互情報、分位点歪度、裾/IQR比、相対IQRを算出した。Q、B、外部BG、候補順位、TPR、対象番号をC特徴に入れていない。

近さは訓練対象だけの標準偏差で標準化したユークリッド距離で測る。候補ごとの期待成績を「全訓練対象の平均」と「Cが近い訓練対象の平均」の混合で予測し、最大の候補を選ぶ。

- 固定版：3近傍、局所比率0.5。
- 調整版：近傍数1/3/5/9、局所比率0.25/0.5/1、加えて局所を使わない基準候補。選択は内側LOTOの平均TPRのみ。
- C-onlyという制限は、対象に応じたゲートの入力に対するもの。基礎攻撃そのものは従来どおり公開BG/Qや評価候補Bを利用する。

### Cから直接分かる構造例

- Team 23：数値8属性の一意値率の平均は0.0380。多くの他対象の約0.77より小さく、強い離散化/値の共有を示唆する。
- Team 19：全属性重複率0.0429。Team 29：0.1115（QI7だけでは0.1125）。C内の重複が直接観測できる。
- Team 24：数値属性間の絶対Spearman相関平均0.6311。Team 17は0.0384。結合関係の強さに大きな違いがある。

これらから加工の特徴は推測できるが、同じ分布は複数の加工法で作れる。上記の特徴から匿名化方式名を断定したり、対応する攻撃が必ず有効だと推測したりはしていない。

## 3. アンサンブル

全候補を875候補内の平均順位に変換して向きを揃えたうえで混合する。元スコアの尺度差を混ぜない。

- 固定対照：全候補平均、家族内/BG平均後の8家族等重み平均、8家族平均の中央値/最大値。
- 学習対照：各訓練foldで各家族の最良設定/BGを1つ選び、8代表の平均/中央値/最大値、訓練上位3/5家族の平均。
- 重み付き：訓練平均TPRのsoftmax（温度0.05を固定）。評価対象の成績は重みに使わない。

全候補平均と家族等重み平均を分けるのは、histogramの1920設定とstable_centersの240設定というグリッド密度差が、そのまま重みになるのを避けるため。

## 4. 主結果：1/2を訓練・評価の両方から除外

'''+table('train_and_test_without1_2')+'''

### 対象別の主比較

'''+ '\n'.join(pt)+'''

8家族代表平均は、基準比でTeam 6/7/10/12/13/18/20が改善、Team 9が0.04低下、残り10対象は同値。固定CゲートはTeam 7/9/10/12で改善し低下なし、調整CゲートはTeam 7/12/24で改善し低下なし。ただし、Cゲートは0.10超の新しい対象を作れなかった。

## 5. 補助結果：全20対象LOTO

'''+table('all20')+'''

### 全20対象で選んだ予測から1/2だけ集計除外

これは訓練側に1/2が入り得るため、主評価の18対象LOTOとは別の表である。

'''+table('without1_2_summary_of_all20')+'''

## 6. 判断と限界

1. 優先するのは18対象の結果。全20平均では簡単なTeam 1/2が強く効き、方針が変わる。全20側での小さな平均差だけを採用理由にしない。
2. C-onlyゲートには少し改善の兆しがあるが、今回の特徴と20件未満の訓練例では有効方式を安定して選ぶ証拠として弱い。Cで方式選択できる可能性全般を否定する試験ではない。
3. 8家族代表平均は単一方式選択の不安定さを和らげた可能性がある。ただしこれは解釈であり、メカニズムを証明していない。今回の14戦略を見た後にこの戦略を採用すれば、戦略選択自体には多重比較が残る。
4. 元グリッドそのものは、過去に公開予備戦の正解を繰り返し確認しながら発展してきた。今回の外側対象の正解を新しいゲート/重みに流していないことは検査したが、完全未使用データに対する独立汎化検証とは呼べない。
5. 対象18個、各対象の正例数約25前後なので、TPRは1件で約0.04動く。CSVの対象bootstrap区間は記述的な感度表示であり、重なったLOTO訓練集合・過去の適応探索・多重比較を補正した有意差ではない。
6. oracleの18/18成功は後出し選択。実行可能な方式として主張しない。

### 次に検証するなら

追加の重いモデル計算は今回行っていない。次段階では、既知の異なる加工設定から疑似Cを生成し、変換強度・乱数・元データを分離した疑似対象でCの指紋と攻撃の効き方を学ぶ方法が候補になる。真の加工法名を当てる分類だけでなく、攻撃損失を直接予測する。加工family丸ごと保留した試験、未知の実データへの最後の保留試験が必要。今回の18対象に合わせて特徴や家族数をさらに調整するだけでは、独立検証は増えない。

## 7. 監査と成果物

- 56 bank、全43,148行列の元SHA-256一致。
- 数値順/文字列順の対象列をIDで整列。5家族のnormal875行は公開された行対応表と照合、残る3家族はrecord_idsと照合。
- 元の対象TPRを各bank先頭/末尾×20対象で再計算し一致（2240件）。
- 最終532個の対象×戦略予測を公式スコア関数で独立再計算し、最大誤差0。
- 外側対象の全成績を大きく改変しても、固定/調整Cゲート・単一方式・家族代表の選択が変わらないことを検査。
- 基礎モデル再実行0、単一スレッド、1bankずつ読み込む低メモリ実行。

主要ファイル：

- `summary.csv`：3種類の評価表示、平均/中央値、≥0.10/>0.10、対応差。
- `primary18_team_comparison.csv`：主評価18対象の比較。
- `heldout_by_target.csv` / `paired_target_deltas.csv`：全対象の結果と基準差。
- `selected_methods_by_target.csv` / `ensemble_training_weights.csv`：外側対象ごとの選択と重み。
- `C_only_features.csv` / `C_only_feature_input_audit.json`：C集約特徴と入力ハッシュ。
- `candidate_catalog.csv`：機械的に採用した候補集合。
- `protocol.json` / `run_experiment.py` / `verify_experiment.py`：事前に固定した条件、再実行コード、独立検証。
- `source_score_audit.json` / `independent_validation.json` / `official_scoring_test.json`：照合結果。

個人別の予測、順位、membership正解は共有用ZIPに含めない。`private_cache/` は内部検証用であり公開対象外。
'''
(R/'Cだけの方式選択とアンサンブル_検証結果.md').write_text(text)
(R/'summary_ja.txt').write_text('主評価はTeam 1/2を訓練・評価から除いた18対象LOTO。TPR@FPR1%、全1049人。\n基準0.004444、C-only固定0.013162、内側調整0.012919。Cゲートの0.10超は0/18。\n訓練側8家族代表の順位平均0.021801、7勝10分1敗、Team13=0.12で0.10超1/18。\n改善は限定的。過去に予備戦正解を見て作られたグリッドでの追加検証であり、独立未使用データの汎化確認ではない。\n全43,148行列のハッシュと最終532予測の公式採点一致。\n')
allowed=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.md','.txt','.csv','.json'] and p.name not in ['delivery_manifest.json']]
files=[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(allowed)]
manifest={'scope':'aggregate results and reproducibility code only','excluded':['private_cache/','per-person predictions','membership truth','source C row data'],'files':files}
(R/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
with zipfile.ZipFile(R/'C単独選択とアンサンブル_集約成果.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in allowed+[R/'delivery_manifest.json']:z.write(p,p.name)
print('REPORT',len(text),'ZIP',(R/'C単独選択とアンサンブル_集約成果.zip').stat().st_size)
