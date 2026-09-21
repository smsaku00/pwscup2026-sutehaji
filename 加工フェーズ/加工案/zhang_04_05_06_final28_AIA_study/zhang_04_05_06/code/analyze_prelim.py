"""予備選公開データの読み取り専用診断。最終3案の生成コードからは呼ばない。"""
import argparse,io,json,zipfile
from pathlib import Path
import numpy as np,pandas as pd

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--result-xlsx',type=Path,required=True)
 ap.add_argument('--attacks-zip',type=Path,required=True)
 ap.add_argument('--truth-zip',type=Path,required=True)
 ap.add_argument('--out',type=Path,required=True)
 args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 scores=pd.read_excel(args.result_xlsx,'総括');rows=pd.read_excel(args.result_xlsx,'攻撃_標的別_AIA')
 names={int(x['コホート番号']):str(x['チーム名']) for _,x in scores.iterrows()}
 diag=rows[rows['標的']==5].copy();diag['attacker_team']=diag['攻撃者'].map(names)
 cols=['攻撃者','attacker_team','R（全体）','R（希少な患者だけ）','p_m','p_c','失点（悪いほう）']
 diag[cols].to_csv(args.out/'prelim_team05_attackers.csv',index=False)
 with zipfile.ZipFile(args.truth_zip) as z:
  truth=pd.read_csv(z.open('正解データ/aia_truth_5.csv'))
 success=(abs(truth.time-10)<=.5)
 time10=truth.assign(hit=success).groupby('label').hit.agg(['sum','count','mean'])
 result={'data':'PRELIM team05 ONLY; not final cohort28','team05_score':scores.loc[scores['コホート番号']==5,['U','MIA 成分','AIA 成分','Anon']].iloc[0].to_dict(),
         'constant10_member':time10.loc['member'].to_dict(),'constant10_control':time10.loc['control'].to_dict(),
         'constant10_R':max(0,(time10.loc['member','mean']-time10.loc['control','mean'])/(1-time10.loc['control','mean'])),
         'attackers20_max_R':float(diag['R（全体）'].max())}
 with zipfile.ZipFile(args.attacks_zip) as outer:
  for attacker in [2,8,23]:
   with zipfile.ZipFile(io.BytesIO(outer.read(f'攻撃提出物/cohort{attacker:02d}_F.zip'))) as inner:
    pred=pd.read_csv(inner.open('F_aia.csv'));p=pred[pred.target_k==5]
   comp=truth.merge(p,on='challenge_row_id');comp['hit']=(comp.time-comp.time_hat).abs()<=.5
   result[f'attacker_{attacker}_success_by_label']=comp.groupby('label').hit.mean().to_dict()
 (args.out/'prelim_team05_baseline.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print('prelim team05 constant10 R',round(result['constant10_R'],4))
if __name__=='__main__':main()
