from pathlib import Path
import time,json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import run_experiment as e
OUT=e.OUT;st=time.perf_counter();audits=[];toprows=[]
assert json.loads((OUT/'run_record.json').read_text())['complete']
for bg in e.BGS:
 d=e.r.data_for(e.u,bg)
 for panel in e.PANELS:
  for t,c in d['cs'].items():
   _,rp,_=e.cachepaths(bg,panel,t);arr=np.load(rp,mmap_mode='r')
   for n in [30,100,1000]:
    seed=20261008+n-1+t;v=e.orig.one_score(d['a'],c,d['b'],panel[1],panel[2],seed)
    ranks=rankdata(v,method='average')/len(v);equal=np.array_equal(ranks,arr[n-1]);assert equal,(bg,panel[0],t,n)
    audits.append(dict(background=bg,panel=panel[0],team=t,N=n,actual_seed=seed,original_checkpoint_seed_ranks_exact=True))
 models=e.loadmodels(bg,1000);_,_,truth=e.u.evaluate(d,models)
 for n in e.NS:
  models=e.loadmodels(bg,n)
  for panel in e.PANELS:
   t=panel[-1];j=d['teams'].index(t);method=e.name(panel);values=models[method][:,j]
   normals=np.flatnonzero(d['normal']);order=np.argsort(-values,kind='stable')[:50]
   for rank,ix in enumerate(order,1):
    full=normals[ix];toprows.append(dict(background=bg,N=n,team=t,method=method,rank=rank,record_id=d['b'].iloc[full].record_id,score=values[ix],membership=int(truth[full,j])))
e.dump(OUT/'extended_seed_original_audit.json',dict(checks=audits,total_checks=len(audits),all_exact=True,seconds=time.perf_counter()-st))
pd.DataFrame(toprows).to_csv(OUT/'focus_top50_records.csv',index=False,float_format='%.17g')
x=pd.read_csv(OUT/'scores_all_targets.csv');x=x[x.fpr==.01];focus=x[((x.team==12)&(x.method==e.name(e.PANELS[0])))|((x.team==3)&(x.method==e.name(e.PANELS[1])))];focus.to_csv(OUT/'focus_scores.csv',index=False,float_format='%.17g')
ranks=pd.read_csv(OUT/'rank_stability.csv');blocks=pd.read_csv(OUT/'seed_block_stability.csv');bounds=pd.read_csv(OUT/'fpr1_boundaries.csv')
lines=['順位集約の反復数比較：10 / 30 / 100 / 1000','', '表示：SPM(P)。SPM=スターター＋予備戦＋本戦の背景A、P=予備戦背景Aのみ。評価B・Cと所属正解は予備戦で固定。スコアは全1049行で公式TPR@FPR1%。','']
for n in e.NS:
 vals=[]
 for t,label in [(12,'k等分'),(3,'ふてねこ')]:
  y=focus[(focus.N==n)&(focus.team==t)].set_index('background').mia_all;vals.append(f'{label} {y[e.BGS[0]]:.6f} ({y[e.BGS[1]]:.6f})')
 lines.append(f'{n}回：'+ ' ／ '.join(vals))
lines+=['','固定条件と再現性','- k等分：numeric8_sexpref_rank_mean__raw__cross0.1。ふてねこ：without_outcomes_rank_median__raw__cross0。','- 各背景で2方式×全20対象を計算。k等分のcross0.1は、そのNでの同方式・自身以外19対象の中央値を引く。','- h=1、relative=0.1、regularizer=1、Cからランダム中心100。seed=20261008+i+対象ID（i=0..999）。各Nは先頭N回を使う入れ子構造。','- A・Cのみでフィットし、Bは順位化のみ。所属正解は各チェックポイントの全対象予測固定後の採点にのみ使用。','- 各回にB全1049行で順位を作り、平均/中央値を計算し、通常875行で再順位化。通常875・希少24・希少以外の既知非共有150。既知非共有マスクは希少との重複13を含む163行で、信頼度0。希少で非既知11行は0.4。通常行は元confidence関数を使用。','- N=10は両背景・両方式・全20対象の既存内部予測とビット単位で完全一致。公式採点との差は0。','- 高速化は公開Cの全候補中心に対するカーネル十分統計の事前計算。全80条件の先頭10seed、および30/100/1000回目のseedを元one_scoreで照合し、1049行の順位が完全一致。','', '順位の変化（100回→1000回）']
for bg in e.BGS:
 for t in [12,3]:
  q=ranks[(ranks.background==bg)&(ranks.team==t)&(ranks.N_from==100)&(ranks.N_to==1000)&(ranks.method==e.name(e.PANELS[0 if t==12 else 1]))].iloc[0]
  lines.append(f'- {bg} / 対象{t}: Spearman={q.spearman:.6f}、上位10={int(q.top10_overlap)}/10、上位20={int(q.top20_overlap)}/20、上位50={int(q.top50_overlap)}/50。')
lines+=['','seedブロック安定性（追加フィットなし）']
for (bg,t,bs),q in blocks.groupby(['background','team','blocksize']):lines.append(f'- {bg} / 対象{t} / 独立した連続{bs}seed×10ブロック: TPR範囲 {q.tpr_001.min():.6f}～{q.tpr_001.max():.6f}、平均 {q.tpr_001.mean():.6f}、標準偏差 {q.tpr_001.std(ddof=1):.6f}。')
lines+=['','解釈上の注意','- 回数を増やすと中心選択の乱数ゆらぎは小さくなるが、TPRが単調に改善する保証はない。順位安定性とTPRを分けて評価する。','- 同じデータ・同じ方式の乱数感度比較であり、新しい独立検証や汎化性能の信頼区間ではない。seedブロックの差もデータ標本の不確実性ではない。','- 予備戦公開正解を使った探索的比較。本戦所属正解は使用していない。','', 'ファイル','- focus_scores.csv / scores_all_targets.csv：対象2チーム / 全20対象の公式採点','- rank_stability.csv：上位10・20・50の重なりとSpearman','- fpr1_boundaries.csv：FPR1%の上下のROC点・閾値・TP/FP数','- seed_block_stability.csv：連続10seed・100seedの各10ブロック感度','- predictions/、prediction_hashes.json：各Nの固定予測とSHA256','- seeds.csv、source/、cache/：再現用コード・seedと中間キャッシュ','- N10_regression_*.json、extended_seed_original_audit.json：既存結果・元実装照合','- timings.json、run_record.json：実行条件・時間','']
metas=[json.loads(p.read_text()) for p in (OUT/'cache').rglob('*_progress.json')]
for bg in e.BGS:
 q=[m for m in metas if m['background']==bg];prep=sum(m['seconds'] for m in q);ext=sum(z['seconds'] for m in q for z in m.get('extensions',[]));lines.append(f'実測カーネル準備＋全1000seed順位計算（{bg}）：{prep+ext:.2f}秒（準備と元10seed照合 {prep:.2f}秒、11～1000回 {ext:.2f}秒）。採点・書出し時間は別。')
lines.append(f'30/100/1000回目seedの追加元実装照合＋レポート処理：{time.perf_counter()-st:.2f}秒。')
(OUT/'結果要約.txt').write_text('\n'.join(lines)+'\n')
e.dump(OUT/'delivery_manifest.json',dict(files=[dict(path=str(p.relative_to(OUT)),sha256=e.r.sha(p),bytes=p.stat().st_size) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'],complete=True))
print('\n'.join(lines),flush=True)
