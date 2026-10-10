from pathlib import Path
import pandas as pd,json,hashlib
O=Path(__file__).parent
I=Path('/workspace/shared/mia_all_background_inventory/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件')
R=Path('/workspace/shared/mia_all_bg_delivery/20261010T025623Z/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件')
D=pd.read_csv(O/'candidate_pool_with_seven_background_summary.csv'); V=pd.read_csv(O/'candidate_pool_seven_backgrounds.csv')
BGS=['starter','prelim','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']
TEAMS={20:'さいふぉん-3',7:'神神公社',8:'CyberMantra',12:'k等分の花嫁',6:'Trailblazer',1:'Ritz',9:'あかぎ団'}
# Choice is by useful distinct computation, never reusing backgrounds as slots.
C=[
(20,'q7_hist5_s100_pairwise__county500__cross0.1','ペアヒストグラム','QI7の属性ペアを5分割し、C/BGの度数比を平均。県・性別較正500と競合補正0.1を加える。','最大値0.222222の主候補。同点のs20は平滑化違いなので別枠にしない。'),
(20,'q7_robust_adaptivek15_h1__county500__cross0','局所帯域KDE密度比','各点の15近傍距離で帯域を変えるKDEのC/BG密度比。QI7・robust変換、県・性別較正500。','0.148148の別方式。5/7背景で0.10超え。'),
(20,'q7_cat4_nnratio_k10__county500__cross0','カテゴリ重視k近傍比','QI7でカテゴリ不一致の距離重みを4とし、BG/Cそれぞれの10近傍平均距離の比を使う。','0.148148の異なる計算法。ピークは1背景。'),
(7,'q13_robust_sex1_diag_reg0.01_gmm4__raw__cross0.1','性別別4成分GMM','13属性・robust変換、性別ごとに対角共分散4成分GaussianをC/BGへ適合。対数密度差に競合補正0.1。','同点0.16のGMMのうち3/7背景で0.10超え。q7 rank/16成分も0.16だが別枠に重複計上しない。'),
(7,'q13_pref_sex_bins40_s5000_logratio__raw__cross0','階層ヒストグラム対数比','13属性を県・性別で条件付け、40分割の度数分布をBGへ強く縮小(s5000)して対数比を計算。','main BGだけ0.16。'),
(7,'source3_all20_temperature2_none4','GMM＋滑らかな競合補正','13属性・robust・性別非分割・対角16成分GMMの順位をlogit化し、他対象への適合と「どこにも属さない」選択肢で補正。','03:10以降の追加候補。GMMベースだが上の4成分方式とはモデル・後処理が違う。'),
(8,'q13_pref_sex_bins10_s10_zscore__raw__cross0.1','13属性・階層頻度差(zscore)','県・性別で条件付けた13属性の10分割ヒストグラムで、BGからの標準化頻度差を合計。縮小s10、競合補正0.1。','0.12で同点、4/7背景が0.10超え。3候補とも同じ階層ヒストグラム系統。'),
(8,'q13_pref_sex_bins10_s10_logratio__raw__cross0.1','13属性・階層対数頻度比','上と同じ属性・10分割・縮小s10だが、標準化差ではなくC/BGの対数頻度比を使う。','zscoreとはスコア式が違う。2/7背景で0.10超え。'),
(8,'q9_pref_sex_bins10_s10_logratio__raw__cross0.1','9属性・階層対数頻度比','9属性へ絞り、県・性別ごとの10分割ヒストグラム対数比を計算。縮小s10、競合補正0.1。','13属性版とは入力属性が違う。同一系統内の代替設定であり独立した第3系統ではない。'),
(12,'q7_all_h2_reg0.01__age100__cross0','全C点中心の直接密度比','QI7をrobust変換し、C全点をカーネル中心にする相対密度比(RuLSIF型)。幅2・正則化0.01、性別・年齢較正100。','最大0.153846で6/7背景が0.10超え。ランダム中心の種違いを重ねず代表化。'),
(12,'q7_rank_adaptivek5_h1__age100__cross0','局所帯域KDE密度比','QI7を順位変換し、各点の5近傍距離で帯域を決めたC/BGのKDE密度比。性別・年齢較正100。','starter BGだけ0.153846。県・性別較正版も同点だが別枠にしない。'),
(12,'conditional_quartile500','他対象順位の四分位較正','他対象にも高く出る傾向を4群へ分け、公開Qの条件付き順位を全体順位へ500件分縮小して補正。','starter+prelimだけ0.153846。通常875人だけの値ではなくall_fixed1049で採点。'),
(6,'q7_robust_lofC_k100__age100__cross0.1','CへのLOF適合度','Cの100近傍に対する局所外れ度スコア。QI7・robust変換、性別・年齢較正100、競合補正0.1。','LOF比ではなくC単体LOF。3/7背景で0.12、他は0.08。'),
(6,'q7_kmeans100_h2_reg0.01__age100__cross0.1','k-means中心の相対密度比','Cの100個のk-means中心を使うRuLSIF型。幅2・正則化0.01、性別・年齢較正100、競合補正0.1。','2/7背景で0.12。次のuLSIFと同じ広義の直接密度比系統。'),
(6,'q7_robust_ulsif_centers300_h4_a0_lambda1__age100__cross0.1','ランダム中心の直接密度比','Cから選んだ300中心のuLSIFでC/BG密度比を直接推定。幅4・相対係数0・正則化1、性別・年齢較正100、競合補正0.1。','prelim+mainだけ0.12。'),
(1,'q7_rank_kde_ratio_h0.5__age100__cross0.1','順位変換KDE密度比','QI7の数値をBG順位へ変換し、半分帯域のKDE密度比に性別・年齢較正100と競合補正0.1を加える。','最大0.84。容易な対象なので他6対象の攻撃力と同列に一般化しない。'),
(1,'sex_point_h0.5','性別別の点KDE密度比','性別ごとに5数値・2カテゴリの候補点密度をC/BGで比較。BG標本数に依存する帯域の0.5倍。','最大0.84。前行と同じKDE系統だが性別分割・変換・較正が違う。'),
(1,'q7_quantile_sex1_kde_witness_h1__raw__cross0','分位整列KDE密度差','Cの数値分布をBGの分位へ整列し、性別分割したKDEのC−BG密度差を使う。','最大0.80。比ではなく差(witness)。3候補とも広い意味ではKDE系統。'),
(9,'q7_pref_sex_bins20_s1_zscore__county500__cross0.1','階層ヒストグラム頻度差','QI7を県・性別で条件付けた20分割ヒストグラムの標準化頻度差。縮小s1、県・性別較正500、競合補正0.1。','最大0.16で6/7背景が0.10超え。age100/rawの同点版を別方式として3枠へ重ねない。'),
(9,'q9_rank_sex1_full_reg0.01_gmm8__raw__cross0.1','性別別8成分GMM','9属性・順位変換、性別ごとに全共分散8成分GaussianのC/BG密度比。競合補正0.1。','0.12はprelim+mainのみ。'),
(9,'q9_rank_lof_ratio_k30__raw__cross0.1','C/BGのLOFスコア差','9属性・順位変換、30近傍LOFのCスコアからBGスコアを引く。競合補正0.1。','IDはratioだが実装は差。starterだけ0.12。他方式として残すピーク候補。'),
]
latest=pd.read_csv(I/'primary_delta_report/01_所属ラベル不使用_対象別差分.csv')
rows=[];vv=[]
for team,m,label,idea,note in C:
 z=D[D.team.eq(team)&D.method.eq(m)]
 if len(z):
  r=z.iloc[0].to_dict(); vals=V[V.team.eq(team)&V.method.eq(m)].copy()
 else:
  z=latest[latest.team.eq(team)&latest.new_panel_method.eq(m)].iloc[0]
  values=[float(z['TPR_'+bg]) for bg in BGS]
  r={'team':team,'team_name':TEAMS[team],'method':m,'method_label':'滑らかな競合補正','canonical_id':'meta_smooth_competition::'+m,'background':z.new_panel_peak_background,'tpr':z.new_panel_peak_TPR,'prelim_bg_tpr':z.same_method_regenerated_prelim_TPR,'min_tpr':min(values),'median_tpr':float(pd.Series(values).median()),'max_tpr':max(values),'background_count_gt_010':sum(round(x,12)>.1 for x in values),'all_best_backgrounds':'|'.join(bg for bg,x in zip(BGS,values) if abs(x-max(values))<1e-12),'source_path':str(I/'primary_delta_report/01_所属ラベル不使用_対象別差分.csv')}
  vals=pd.DataFrame([{'team':team,'team_name':TEAMS[team],'canonical_id':r['canonical_id'],'method':m,'background':bg,'tpr':value,'members':int(z.members),'source_path':r['source_path']} for bg,value in zip(BGS,values)])
 r.update({'plain_method':label,'key_idea_ja':idea,'selection_note':note,'training_category':'no_membership_labels','inference':'all20','population':'all_fixed','n':1049,'fpr':.01,'status':'予備戦の再生成実験結果。公式提出・採点記録であることはこの表では未確認。'})
 for bg in BGS:r['TPR_'+bg]=float(vals[vals.background.eq(bg)].tpr.iloc[0])
 rows.append(r);vv.append(vals)
F=pd.DataFrame(rows);F['candidate_order']=F.groupby('team',sort=False).cumcount()+1
columns=['team','team_name','candidate_order','plain_method','method','canonical_id','max_tpr','all_best_backgrounds','prelim_bg_tpr','min_tpr','median_tpr','background_count_gt_010','key_idea_ja','selection_note','training_category','inference','population','n','fpr','status','source_path']+['TPR_'+bg for bg in BGS]
F[columns].to_csv(O/'recommended_three_methods_per_team.csv',index=False,encoding='utf-8-sig')
pd.concat(vv).to_csv(O/'recommended_three_methods_seven_backgrounds.csv',index=False,encoding='utf-8-sig')
# Membership-label learned category stays separate.
L=pd.read_csv(I/'primary_delta_report/02_OOF学習_対象別差分.csv');L=L[L.team.isin(TEAMS)];L['team_name']=L.team.map(TEAMS);L.to_csv(O/'membership_label_OOF_optional_separate.csv',index=False,encoding='utf-8-sig')
# Related variants that should not be concealed or counted as independent families.
alt={20:['q7_hist5_s20_pairwise__county500__cross0.1'],7:['q13_robust_sex0_diag_reg0.01_gmm4__raw__cross0','q7_rank_sex1_full_reg0.01_gmm16__county500__cross0'],8:['q13_pref_sex_bins10_s1_zscore__raw__cross0.1','q13_pref_sex_bins10_s100_logratio__raw__cross0.1'],12:['quasi7_seed20261014__age100__cross0','q7_robust_ulsif_centers300_h1_a0_lambda0.0001__age100__cross0'],9:['q7_pref_sex_bins20_s1_zscore__age100__cross0.1','q7_pref_sex_bins20_s1_zscore__raw__cross0.1']}
A=pd.concat([D[D.team.eq(t)&D.method.isin(mm)] for t,mm in alt.items()]);A.to_csv(O/'related_variants_not_extra_families.csv',index=False,encoding='utf-8-sig')
lines=['# 継続が似る7チーム：予備戦MIAの有望な約3方式','',
'## 比較条件と選び方',
'- 主表は所属ラベルを学習に使わない方式、20対象推論、all_fixed 1,049人、公式採点式TPR@FPR=1%。mainは背景資料名で、本戦TPRではない。',
'- 7背景はstarter / prelim / mainの非空組合せ。同じデータを共有する感度条件であり、独立した7回の外部検証ではない。',
'- 同じcanonical_idのBG違いは1方式にまとめた。出所で同一と証明された別名は元表の統合を保持。値が一致しただけでは同一方式と見なしていない。',
'- ここでの「3候補」は同系統の平滑化・較正違いで埋めず、最大TPRと異なる計算アイデアを重視した例。純粋な3設定順位ではない。数値順の厳密上位3設定はstrict_top3_distinct_parameter_recipes.csvに別保存した。',
'- Ritzは3種とも広義のKDE。CyberMantraは3種とも階層ヒストグラム。TrailblazerはLOF＋2種の直接密度比。神神公社は2種のGMMベース＋ヒストグラム。3個の独立系統が揃ったと誇張しない。',
'- いずれも正解を見た後の探索的な候補選定であり、未知の本戦で使える対象別選択規則を検証したものではない。公式採点「式」を用いた再生成実験であって、公式に提出・採点された証拠とは区別する。',
'- 03:10以降の追加主評価は20対象すべてで既存最高値を更新せず。ただし神神公社の滑らかなGMM競合補正など、同点の別候補は追加されている。','',
'## 7対象の3候補','']
for t,name in TEAMS.items():
 lines += [f'### {name}（予備戦Team{t}）','', '|候補|計算アイデア|最大TPR|最良BG（全て）|7BG範囲|0.10超え|','|---|---|---:|---|---|---:|']
 for r in F[F.team.eq(t)].to_dict('records'):
  lines.append(f"|{int(r['candidate_order'])}|{r['plain_method']}|{r['max_tpr']:.6f}|{r['all_best_backgrounds']}|{r['min_tpr']:.6f}〜{r['max_tpr']:.6f}|{int(r['background_count_gt_010'])}/7|")
 lines+=['']
 for r in F[F.team.eq(t)].to_dict('records'):
  lines += [f"{int(r['candidate_order'])}. `{r['method']}`",f"   - {r['key_idea_ja']}",f"   - {r['selection_note']}"]
 lines += ['']
lines+=['## 同系統の補足と見方','',
'- さいふぉん-3：ペアヒストグラムは6/7背景で0.10超え、中央値0.222222。s20へ変えても同じ最大・中央値。対象20に強いが、同じ固定レシピが0.10を超える非容易対象は20だけという既存分析であり、万能な候補ではない。',
'- 神神公社：最大0.16の攻めたGMMとは別に、q13_robust_sex0_diag_reg0.01_gmm4__raw__cross0 は全7背景0.12。近傍24設定中12設定も全7背景0.10超えで、安定側の比較対象として残す価値がある。',
'- CyberMantra：主比較の0.10超えは6設定・12設定BG行しかなく、全て階層ヒストグラム。3候補は同じ原理の異なる式・属性レシピであり、3つの強い独立手法は見つかっていない。s10 zscoreは4/7背景0.12で、s1やs100の単一ピーク優先より同点内で扱いやすい。',
'- k等分の花嫁：quasi7_seed20261014__age100__cross0 も6/7背景で0.10超え、中央値0.153846。ただし属性・乱数中心の密度比系統なので主表の全中心型と同じ枠の代替として記録。直接uLSIF q7_robust_ulsif_centers300_h1_a0_lambda0.0001__age100__cross0 も最大0.153846、4/7背景0.10超え。',
'- Trailblazer：主比較の0.10超えは4設定・7設定BG行。最良0.12は25会員中3人相当で、LOF以外の2候補は最低0まで落ちる。神神公社の全7背景0.12や、さいふぉん-3の0.222222とは証拠の厚さが違う。',
'- Ritz：最大0.84だが容易な対象。弱い対象へそのまま一般化しない。',
'- あかぎ団：階層ヒストグラムが主候補。county500版とraw版はいずれも6/7背景0.10超えで最大0.16。age100版も同点0.16だが4/7。同点3較正版を3方式と呼ぶより、GMM・LOFの0.12を別の攻めた候補として見せる。','',
'## 所属ラベルOOF学習は別カテゴリ','',
'下記は公開所属正解を学習に用いる比較。主表へ混ぜない。選出設定の数値再現差・多数設定からの事後選択も残る。','',
'|対象|同カテゴリの既報/追加最大TPR|追加代表のBG範囲|0.10超え|','|---|---:|---|---:|']
for t,name in TEAMS.items():
 r=L[L.team.eq(t)].iloc[0];lines.append(f"|{name}|{max(r.new_panel_peak_TPR,r.previous_0310_category_best_TPR):.6f}|{r.same_method_7BG_min:.6f}〜{r.new_panel_peak_TPR:.6f}|{int(r.same_method_7BG_above010)}/7|")
lines += ['', 'あかぎ団のOOF 0.20はstarterの1背景だけで、選出設定の7BG中央値は0。主表のヒストグラム0.16を0.20へ置き換えて「同条件で改善」と報告しない。','',
'## 出典・検証','',
'- 03:10凍結順位：team_rankings_0310UTC/primary_all20_all1049_strictly_gt_010.csv、top10_per_team_primary_all20_all1049.csv。',
'- 全BG値は02:56:23UTC封印版の各方式元CSVから再抽出。source_manifest.jsonに実ファイル絶対パスとSHA-256を保存。予測生成・学習・採点の再実行はしていない。',
'- 追加比較：primary_delta_report/01_所属ラベル不使用_対象別差分.csv、02_OOF学習_対象別差分.csv、10_追加パネル_対象別上位10方式.csv、README.md。',
'- 既存解釈：promising_methods_0310UTC/有望方式の考察と公開BG一致の影響.md、candidate_stability_summary.csv。',
'- 計算式は封印版recovered_sources/本戦/攻撃フェーズ/Tashiro/20261008_MIA全体_朝9時までの方式探索/{距離と分布の探索,階層ヒストグラムの探索,混合分布と局所密度の探索,直接密度比の探索,中心配置を安定させた密度比,他対象との競合を滑らかに補正,共通}.py、20261007_MIA全体改善の検討/追加検討.py、available/original/general.pyと対応実行ラッパーを参照。',
'- 純粋な設定順位のファイルは03:10版だけ。本文の3候補には後のsmooth追加を反映。関連変種を別ファイルに保存し、削除や改変していない。','']
(O/'予備戦7チーム_有望3方式レビュー.md').write_text('\n'.join(lines))
assert F.groupby('team').size().eq(3).all()
assert not F.duplicated(['team','canonical_id']).any()
assert len(pd.concat(vv))==147
assert F.training_category.eq('no_membership_labels').all()
manifest=json.loads((O/'source_manifest.json').read_text())
manifest['additional_summary_sources']=[{'source_path':str(I/p),'sha256':hashlib.sha256((I/p).read_bytes()).hexdigest()} for p in ['primary_delta_report/01_所属ラベル不使用_対象別差分.csv','primary_delta_report/02_OOF学習_対象別差分.csv','primary_delta_report/10_追加パネル_対象別上位10方式.csv','promising_methods_0310UTC/有望方式の考察と公開BG一致の影響.md']]
manifest['validation']={'selected_rows':21,'selected_BG_rows':147,'distinct_recipe_per_team':3,'regenerated_experiments':False,'protocol':'no_membership_labels|all20|all_fixed1049|FPR.01','unverified_official_submission':True}
(O/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(F[['team','plain_method','max_tpr','min_tpr','background_count_gt_010']].to_string(index=False))
