"""保存結果の18対象比較を再集計し、任意で公式ROCの独立再採点をする。"""
from __future__ import annotations
import argparse,json,time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import pandas as pd
import 共通 as u

def source_tables():
 for path in sorted(u.HERE.rglob('対象別TPR.csv')):
  # 再集計自身の出力と失敗記録は入力としない。
  if any(s.startswith('再集計_') or s=='失敗記録' for s in path.relative_to(u.HERE).parts):continue
  record=path.parent/'実行記録.json'
  if record.exists():yield path

def verify_record_hashes():
 rows=[]
 for path in sorted(u.HERE.rglob('実行記録.json')):
  if any(s.startswith('再集計_') or s=='失敗記録' for s in path.relative_to(u.HERE).parts):continue
  record=json.loads(path.read_text())
  for kind in ('input_sha256','support_code_sha256','output_sha256'):
   for name,expected in record.get(kind,{}).items():
    source=Path(name) if kind!='output_sha256' else path.parent/name
    actual=u.sha(source) if source.is_file() else None
    rows.append(dict(record=str(path.relative_to(u.HERE)),kind=kind,file=str(source),matches=actual==expected))
  if 'code_sha256' in record:
   # 実行器の対応は入力ハッシュまたは実験方針に残してある。
   rows.append(dict(record=str(path.relative_to(u.HERE)),kind='recorded_code_hash',file='record_value',matches=len(record['code_sha256'])==64))
 return pd.DataFrame(rows)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path);parser.add_argument('--verify-ranks',action='store_true');parser.add_argument('--all-backgrounds',action='store_true')
 args=parser.parse_args();started=time.perf_counter();now=datetime.now(ZoneInfo('Asia/Tokyo'));out=args.output_dir or u.HERE/('再集計_'+now.strftime('%Y%m%d_%H%M%S'));assert not out.exists()
 rows=[];inputs={};sources=[]
 for path in source_tables():
  inputs[str(path)]=u.sha(path);table=pd.read_csv(path);sources.append(str(path.parent.relative_to(u.HERE)))
  assert set(['method','team','fpr','mia_all']).issubset(table.columns)
  for (method,point),part in table.groupby(['method','fpr']):
   assert part.team.nunique()==20 and len(part)==20
   score=part.set_index('team').mia_all
   rows.append(dict(source=str(path.parent.relative_to(u.HERE)),method=method,fpr=point,all20=float(score.mean()),other18=float(score.drop([1,2]).mean())))
 aggregate=pd.DataFrame(rows);assert not aggregate.empty
 hashes=verify_record_hashes();assert hashes.matches.all(),hashes[~hashes.matches].to_dict('records')
 rank_checks=[]
 if args.verify_ranks:
  data=u.setup()
  for path in sorted(u.HERE.rglob('内部順位キャッシュ.npz')):
   if any(s.startswith('再集計_') or s=='失敗記録' for s in path.relative_to(u.HERE).parts):continue
   if not args.all_backgrounds and len(path.relative_to(u.HERE).parts)>2 and path.parent.name not in ('full','採点'):continue
   csv=path.parent/'対象別TPR.csv'
   if not csv.exists():continue
   inputs[str(path)]=u.sha(path);z=np.load(path);assert list(z['teams'])==data['teams']
   if 'record_ids' in z:assert np.array_equal(z['record_ids'],data['b'].loc[data['normal'],'record_id'].to_numpy())
   models={str(name):matrix for name,matrix in zip(z['names'],z['scores'])};table,error,_=u.evaluate(data,models)
   saved=pd.read_csv(csv);merged=table.merge(saved,on=['method','team','fpr'],suffixes=('_new','_saved'),validate='one_to_one')
   maximum=float(np.max(np.abs(merged.mia_all_new-merged.mia_all_saved)));assert maximum<1e-12 and len(merged)==len(table)==len(saved)
   rank_checks.append(dict(source=str(path.parent.relative_to(u.HERE)),models=len(models),score_values=len(table),saved_value_max_error=maximum,official_tpr_max_error=error))
   u.log(stage='independent_rank_verification',source=str(path.parent.relative_to(u.HERE)),maximum_error=maximum,models=len(models))
 for p,h in inputs.items():assert u.sha(p)==h,p
 out.mkdir();aggregate.to_csv(out/'18対象の全方式.csv',index=False,float_format='%.17g');hashes.to_csv(out/'既存記録のハッシュ照合.csv',index=False)
 if rank_checks:pd.DataFrame(rank_checks).to_csv(out/'順位からの独立再採点.csv',index=False,float_format='%.17g')
 best=aggregate[aggregate.fpr==.01].sort_values('other18',ascending=False).groupby('source',sort=False).head(1)
 best.to_csv(out/'各実験の最大値.csv',index=False,float_format='%.17g')
 lines=['# 保存結果の再集計','',f'実行時刻: {now.isoformat()}。主指標は1・2を除く18対象のTPR@FPR1%。','',
        '各実験の最大値は探索の診断値であり、採用候補や独立検証の成績ではない。背景抽出・人物分割・乱数の各最大値も同様に扱う。','',
        '| 実験・背景条件 | 18対象最大値 | 設定 |','|---|---:|---|']
 for _,row in best.iterrows():lines.append(f'| {row.source} | {row.other18:.6f} | {row.method} |')
 (out/'説明.md').write_text('\n'.join(lines)+'\n')
 record=dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),seconds=time.perf_counter()-started,input_sha256=inputs,code_sha256=u.sha(__file__),source_tables=len(sources),aggregated_settings=len(aggregate),
             hash_checks=len(hashes),rank_checks=rank_checks,main_truth_used=False,output_sha256={p.name:u.sha(p) for p in out.iterdir()},note='既存表からの再集計。verify-ranksでは内部順位から公式ROCを独立再計算。学習を再実行する場合は各実験スクリプトに新しいoutput-dirを渡す。')
 (out/'実行記録.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n');u.log(stage='aggregation_complete',output=str(out),source_tables=len(sources),seconds=record['seconds'])

if __name__=='__main__':main()
