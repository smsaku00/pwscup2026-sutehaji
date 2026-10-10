import csv,json,hashlib
from pathlib import Path
import openpyxl
out=Path('/workspace/shared/攻撃フェーズ/Tashiro/MIA全体/continuity_attack_review_20261010/decision_review_20261010')
base=Path('/workspace/shared/攻撃フェーズ/Tashiro/MIA全体')
pre=Path('/workspace/shared/mia_recovered_validation/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/private_inputs/公開用結果表_20260909.xlsx')
rbase=Path('/workspace/shared/mia_round4_publish/stage/攻撃フェーズ/Tashiro/MIA全体/rounds')
u=list(csv.DictReader((base/'本戦公式有用性_20261010/main_utility_scores.csv').open()))
r4={int(r['team']):r for r in csv.DictReader((rbase/'20261010_第4回方式照合/column_comparison.csv').open())}
s=openpyxl.load_workbook(pre,read_only=True,data_only=True)['総括']; p={int(s.cell(r,1).value):(s.cell(r,3).value,s.cell(r,13).value,r) for r in range(2,24)}
mp={0:4,1:8,2:18,3:29,4:13,5:7,6:12,7:6,8:26,9:2,10:24,11:None,12:1,13:9,14:10,15:20,16:17,17:21,22:3,25:22,26:23,27:19}
props={
0:('現行維持／探索GMM8候補','q13_rank_sex1_full_reg0.01_gmm8__raw__cross0.1','探索的GMM候補。保守監査では現行維持。切替優位が未確認で採用・枠配分未決。'),
1:('県×性別ヒストグラム','q13_pref_sex_bins10_s10_zscore__raw__cross0.1','予備戦比較から分布の度数差を使う候補。P01からの改善を試す案で、本戦への移行効果は未確定。'),
2:('KDE witness','q13_robust_kde_witness_h2__raw__cross0','密度差に着目した候補。県性別較正P01とは異なる手掛かりを試す案。'),
3:('県×性別ヒストグラム','q7_pref_sex_bins20_s1_zscore__age100__cross0.1','予備戦比較からヒストグラム系列を候補化。背景行・値の再利用に関する従来仮説を別方式で検討。'),
4:('現行整列最近傍を維持／BGSPM比較','q7_median_iqr_sex1_dcr_k1__raw__cross0','現行の性別内中央値/IQR整列・最近傍を維持する案。BGSPMは比較候補で置換確定ではない。'),
5:('性別分割GMM4','q13_robust_sex1_diag_reg0.01_gmm4__raw__cross0.1','既存評価のピーク0.16を得た候補。性別非分割・cross0のGMMはTPR 0.12で安定。ピークは本戦予測値ではない。'),
6:('全C中心の直接密度比','q7_all_h2_reg0.01__age100__cross0','10/30/100/1000回の反復平均・中央値で得点低下があり、反復順位集約から全C決定論方式へ切替提案。'),
7:('C側LOF・k100','q7_robust_lofC_k100__age100__cross0.1','既存LOF候補の予備戦TPRは0.12、通常者TPRも0.12。背景条件間の範囲0.08〜0.12を踏まえて候補化。本戦未確認。'),
8:('性別非分割GMM2','q7_robust_sex0_diag_reg0.01_gmm2__age100__cross0.1','予備戦比較で混合分布系列を候補化。現行固定幅KDEとの差は本戦未検証。'),
9:('P01保存列を完全保持','P01_saved_column_1049_exact_preserve','ユーザー指定。保存P01の1,049行列をそのまま保持し、通常者・希少者を含め変更しない。'),
10:('中央値/IQR整列KDE密度比','q7_median_iqr_sex0_kde_ratio_h1__raw__cross0.1','現行固定幅KDEに位置・幅の整列を加える案。予備戦比較の根拠であり本戦改善は未確定。'),
11:('採用未決：現行分位点KDE／探索P01','P01','予備戦対応データなし。保存済み分位点整列KDEを基準にP01を探索比較する案で、切替確定ではない。'),
12:('性別点密度 h0.5','sex_point_h0.5','Ritz向け既存評価を踏まえた候補。P01の継続より点密度を試す案で、本戦改善は未確定。'),
13:('県×性別ヒストグラム','q7_pref_sex_bins20_s1_zscore__county500__cross0.1','予備戦比較を踏まえ、県性別較正を残してヒストグラム系列へ切替提案。'),
14:('性別内中央値/IQR整列KDE密度比','q7_median_iqr_sex1_kde_ratio_h1__age100__cross0.1','年齢較正を残し分布整列を使う候補。第4回Gen-LRAからの切替案で本戦優位は未確定。'),
15:('二変量ヒストグラム','q7_hist5_s100_pairwise__county500__cross0.1','既存比較から属性対の度数差を使う候補。第4回可変幅KDEの探索的選択を見直す。'),
16:('LOF比・k100','q7_robust_lof_ratio_k100__age100__cross0','現行分位点整列KDEとは異なる局所密度比を試す候補。本戦性能の保証はない。'),
17:('採用未決：P01維持／探索固定幅KDE','q7_robust_kde_ratio_h1__raw__cross0.1','予備戦Cと公式得点が欠損し移行根拠が弱い。固定幅KDEは探索候補に留める。'),
22:('C側LOF・k100','q13_robust_lofC_k100__raw__cross0','既存比較で局所外れ値を使う候補。第4回可変幅KDEからの切替案で本戦性能は未確定。'),
25:('現行維持／探索ヒストグラム候補','q13_pref_sex_bins40_s100_logratio__raw__cross0.1','探索的ヒストグラム候補。保守監査では現行維持。切替優位が未確認で採用・枠配分未決。'),
26:('最遠点100中心の直接密度比','q7_farthest100_h2_reg1__raw__cross0','離散化解除を踏まえ、連続値を扱う決定的中心の密度比を比較。予備戦TPR 0.12、通常者TPR 0.12。本戦未確認。'),
27:('県×性別ヒストグラム','q7_pref_sex_bins20_s1_logratio__raw__cross0','既存比較に基づく度数比候補。第4回の探索的年齢較正Gen-LRAを見直す。')}
def rname(x):
 if x=='P01':return 'P01（Gen-LRA7・県×性別較正500）'
 if x=='GenLRA7_age100':return 'Gen-LRA7・性別×年齢較正100'
 if 'adaptive' in x:return '可変幅KDE密度比'
 if 'quantile' in x:return '性別内分位点整列KDE'
 if 'dcr' in x:return '性別内中央値/IQR整列・最近傍k1'
 return '固定幅KDE密度比'
headers=['本戦チーム番号','予備戦チーム番号','チーム名','本戦有用性U','現在の採用予定（第5回向け案）','第4回までの方式','採用予定の理由・留保','予備戦有用性U','予備戦匿名性Anon','状態','候補方式ID','第4回方式ID','本戦U出典','予備戦原本セル','第4回検証状況']
rows=[]
for a in u:
 t=int(a['cohort'])
 if t not in mp:continue
 pid=mp[t]; pu,pa,pr=p[pid] if pid else (None,None,None)
 label,recipe,why=props[t]
 status='提案・未提出／最終採用未承認'
 if t in [0,11,17,25]:status='採用未決定・現行維持／探索候補'
 if t==9:status='ユーザー指定の維持（変更なし）'
 if t==4:status='現行維持案・BGSPM比較未決'
 name=a['team']; name='全部生成 / JustGenerateIt' if t==14 else name
 hist='第1回：Gen-LRA（概要）；第2回：seo-02（概要）；第3回：P01（概要）；第4回：'+rname(r4[t]['recipe_id'])
 rows.append([t,pid if pid is not None else '対応なし',name,float(a['U']),label,hist,why,pu if pu is not None else ('欠損' if pid else '対応なし'),pa if pa is not None else ('欠損' if pid else '対応なし'),status,recipe,r4[t]['recipe_id'],'main_utility_scores.csv / cohort='+str(t)+' / 2026-09-24T23:55:23Z',f'総括!C{pr},M{pr}' if pr else '予備戦対応なし',r4[t]['verification']])
assert len(rows)==22 and [r[0] for r in rows]==sorted(mp)
with (out/'全22チーム_第5回向け採用判断案.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(headers);w.writerows(rows)
notes=[['項目','内容'],['位置づけ','2026-10-10時点の第5回向け判断案。全22対象の新方式採用を意味しない。候補は未提出、最終承認・枠配分は未決。'],['方式履歴','第1回Gen-LRA、第2回seo-02、第3回P01は保存rounds/README.mdの概要記録。第1・2回の生成原本は今回未回収。第3回はP01保持8対象の第4回非希少1,025セル一致を確認したが、他14対象は今回第3回列の独立再検証なし。第4回は対象別設定・保存列の照合。過去4回が同一方式という意味ではない。第4回方式の検証は通常875人。既知非会員150人、希少24人の処理を含む全1,049行とは区別。希少処理は回別に異なる。'],['本戦U','公式加工フェーズ採用提出の保存スナップショット。小数4桁表示。攻撃TPR・第5回予測ではない。'],['予備戦U・Anon','公開用結果表_20260909.xlsx の総括!C列・M列を原本から読取。表示は小数6桁、CSVは元精度。Anonは大きいほど匿名性が高い。'],['欠損','bot035（予備戦21）の総括!C18/M18は空欄。0点に置換しない。HiddenMetricsは予備戦対応なし。'],['評価の限界','候補比較・ピークは予備戦等の既存評価に由来し、本戦未知真値に対する性能証明ではない。新実験は実施していない。'],['反復集約','10/30/100/1000回の反復で低下した順位平均・中央値は最新案から除外。反復検討対象は本戦6（k等分）と22（ふてねこ）。対象6は全C、対象22はLOFの決定論的候補。TrailblazerのLOF選定理由とは別。'],['Haruna','本戦9は保存P01の1,049行をそのまま保持するユーザー指定。今回ファイルに予測列自体は含めず、指示を記録。'],['本戦数値ソース',str(base/'本戦公式有用性_20261010/main_utility_scores.csv')],['予備戦数値ソース',str(pre)],['第4回ソース',str(rbase/'20261010_第4回方式照合/column_comparison.csv')],['第1〜3回ソース',str(rbase/'README.md')],['候補判断ソース','2026-10-10の会話で整理されたSol6.1 Ultra監査・反復評価後の最新候補指定。今回の作業は既存値と履歴の照合・一覧化。']]
(out/'data.json').write_text(json.dumps({'headers':headers,'rows':rows,'notes':notes},ensure_ascii=False,indent=2))
manifest={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [pre,base/'本戦公式有用性_20261010/main_utility_scores.csv',rbase/'20261010_第4回方式照合/column_comparison.csv',rbase/'README.md']}
(out/'source_hashes.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
for r in rows:print(' | '.join(str(v) if not isinstance(v,float) else (f'{v:.4f}' if i==3 else f'{v:.6f}') for i,v in enumerate(r[:9])))
