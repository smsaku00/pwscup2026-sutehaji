"""昨日の保存済み対象別TPRを、高得点1・2を除いて比較する。"""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'20261007_MIA全体改善の検討'
CAL=HERE.parent/'20261008_MIA全体_較正の広範比較'
OUT=HERE/'昨日の再評価結果'
SOURCES=[
 '結果/予備戦_対象別比較.csv',
 '追加結果_最終版/予備戦_追加対象別比較.csv',
 '参照結果_拡張版/参照_対象別比較.csv',
 '属性部分集合結果/部分集合_対象別比較.csv',
 '背景再標本化結果/背景再標本化_対象別比較.csv',
 '学習結果_順位校正版/学習_対象別比較.csv',
 '非会員較正結果/非会員較正_対象別比較.csv',
 '固定候補の感度結果/感度検証_対象別比較.csv',
 '非会員較正の感度結果/較正感度_対象別比較.csv',
 '低FPR結果/低FPR_対象別比較.csv',
 '非会員較正の感度結果/較正低FPR_対象別比較.csv']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def condition(method):
 if '__' in method:return method.split('__',1)[0]
 if method.startswith(('half0_', 'half1_')):return method.split('_',1)[0]
 return 'full'

def main():
 if OUT.exists():raise FileExistsError('履歴は上書きしない。')
 refs=pd.read_csv(CAL/'結果/対象別TPR.csv')
 refs=refs[(refs.method=='flat_pref_sex_s500')&(refs.fpr==.01)]
 reference={c:x.set_index('team').mia_all for c,x in refs.groupby('condition')}
 hashes={str(CAL/'結果/対象別TPR.csv'):sha(CAL/'結果/対象別TPR.csv')}
 rows=[];details=[]
 for source in SOURCES:
  p=OLD/source; hashes[str(p)]=sha(p)
  x=pd.read_csv(p)
  if 'fpr' in x:x=x[np.isclose(x.fpr,.01)]
  for method,g in x.groupby('method',sort=True):
   assert len(g)==20 and g.team.is_unique,(source,method)
   s=g.set_index('team').mia_all
   other=s.drop([1,2]);c=condition(method)
   row=dict(source=source,method=method,condition=c,all20=float(s.mean()),other18=float(other.mean()),
            largest_target=float(other.max()),nonzero_targets18=int((other>0).sum()),
            includes_highscore_labels_in_training='学習結果' in source,
            reference_available=c in reference)
   if c in reference:
    d=other-reference[c].loc[other.index];positive=d[d>1e-12].sort_values(ascending=False)
    row.update(delta_vs500=float(d.mean()),wins_vs500=int((d>1e-12).sum()),losses_vs500=int((d< -1e-12).sum()),
               gain_after_removing_best2=float(d.drop(positive.index[:2]).sum()),
               largest_gain_share=float(positive.iloc[0]/positive.sum()) if len(positive) else 0.)
   rows.append(row)
   for team,value in s.items():details.append(dict(source=source,method=method,condition=c,team=int(team),mia_all=float(value),primary=int(team) not in (1,2)))
 table=pd.DataFrame(rows)
 assert len(table[table.condition=='full'])>100
 OUT.mkdir()
 table.to_csv(OUT/'昨日の方式_18対象再評価.csv',index=False,float_format='%.17g')
 pd.DataFrame(details).to_csv(OUT/'昨日の方式_対象別.csv',index=False,float_format='%.17g')
 full=table[table.condition=='full'].sort_values(['other18','wins_vs500'],ascending=False)
 full.to_csv(OUT/'全背景の比較.csv',index=False,float_format='%.17g')
 for p,h in hashes.items():assert sha(Path(p))==h
 record=dict(source_sha256=hashes,code_sha256=sha(Path(__file__)),configurations=len(table),full_configurations=len(full),
             note='旧学習結果の18対象再採点でも、旧モデルは対象1・2の正解を学習している。新規学習では除外する。')
 (OUT/'実行記録.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 print(full[['source','method','all20','other18','wins_vs500','losses_vs500','gain_after_removing_best2']].head(18).to_string(index=False))
 print('saved configurations',len(table),'full',len(full))

if __name__=='__main__':main()
