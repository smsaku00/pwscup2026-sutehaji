"""既存MIA全体の対象別値を統合する。真値/個人行/提出回答は出力しない。

ルート .venv/bin/python -B 集計.py --root <PWSローカル> --output-dir <新しい保存先>
--output-dir の既存評価成果物は上書きしない。GitHubへの書込みは行わない。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import platform
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import sys

from official_audit import band, load_truth, load_official
import oct07_results
import oct08_results
import historical_results
import historical_virtual
import historical_atk

def write_csv(path, rows, fields):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader();w.writerows(rows)

def hash_file(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def change_from_p01(r):
    m=r['method']
    if m=='Q_prefecture_sex_shrink500_cross01':return 'P01と同じ方式。背景・分割条件はcondition参照'
    if m=='Q_prefecture_sex_shrink100_cross01':return 'Q県×性別の縮約500→100、Gen7とcross0.1は維持'
    if m=='mix_county200_age100_w0.1':return 'Q県×性別200 CDF90%＋Q性別×年齢100 CDF10%、Gen7/cross0.1維持'
    if 'nn' in m or 'dcr' in m:return 'Gen-LRA/Q較正に代えて近傍距離比等。属性・尺度・k・crossはparams参照'
    if r['evaluation_class']=='B':return '公開予備戦所属ラベルによる分割学習を追加。評価対象ラベルはsplitに従い保留'
    if 'point' in m or 'kde' in m or 'density' in m:return '密度推定/密度比へ変更。Gen局所追加尤度と異なる推定器'
    if m=='global7':return 'Q県×性別較正500とcross0.1を外したGen-LRA7'
    if m.startswith(('global9','sex9')):return '数値BMI/FPGを追加した9属性Gen-LRA。性別分割/較正/crossは設定名参照'
    return 'P01に対する属性・参照・較正・推定器/係数の既存変更。方式名、attributes、params、background参照'

def integrate(root):
    truth=load_truth(root)
    counts={team:int(y.sum()) for team,y in truth.items()}
    allrecords=[];notes=[]
    for mod in (oct07_results,oct08_results,historical_results,historical_virtual,historical_atk):
        records,entrynotes=mod.gather(root)
        allrecords.extend(records);notes.extend(entrynotes)
    # 最終報告にTeam12を含めない。学習器の入力対象は変更しない。
    selected=[r for r in allrecords if r['team'] not in (5,12)]
    grouped=defaultdict(list)
    for r in selected:
        r['tpr']=float(r['tpr']); assert math.isfinite(r['tpr']) and 0<=r['tpr']<=1
        assert r['team'] in counts
        if r.get('members'):assert int(r['members'])==counts[r['team']],(r['source'],r['team'],r['members'])
        r['members']=counts[r['team']]
        r.setdefault('selection','fixed_rule')
        if r['selection']=='事前固定方式':r['selection']='fixed_rule'
        grouped[(r['method'],r['condition'],r['evaluation_class'],r['selection'],r['team'])].append(r)
    canonical=[];conflicts=[]
    for key, aliases in sorted(grouped.items()):
        vals=[r['tpr'] for r in aliases]
        if max(vals)-min(vals)>1e-12 or len({band(v) for v in vals})>1:
            conflicts.append({'key':key,'values':vals,'sources':[r['source'] for r in aliases]})
            # 同名で異なる計算履歴を混ぜない。
            for r in aliases:
                r=r.copy();r['method']+='@'+hashlib.sha256(r['source'].encode()).hexdigest()[:8]
                canonical.append(r)
            continue
        best=max(aliases,key=lambda x:('公式再採点' in x.get('verification',''),len(x.get('params',''))))
        r=best.copy(); r['sources']=sorted({x['source'] for x in aliases})
        r['aliases']=sorted({x.get('original_method',x['method']) for x in aliases})
        canonical.append(r)
    return canonical,notes,conflicts,len(allrecords),len(selected)

def exports(root,out):
    rows,notes,conflicts,raw_count,report_count=integrate(root)
    p01={r['team']:r['tpr'] for r in rows if r['method']=='Q_prefecture_sex_shrink500_cross01' and r['condition']=='full'}
    assert len(p01)==19,p01
    sourcepaths=sorted({s for r in rows for s in r.get('sources',[r['source']])})
    sourceids={s:f'S{i:04d}' for i,s in enumerate(sourcepaths,1)}
    cfgkeys=sorted({(r['method'],r['condition'],r['evaluation_class'],r['selection']) for r in rows})
    cfgids={k:f'C{i:05d}' for i,k in enumerate(cfgkeys,1)}
    bycfg=defaultdict(list)
    for r in rows:
        k=(r['method'],r['condition'],r['evaluation_class'],r['selection'])
        r['condition_id']=cfgids[k]
        r['band']=band(r['tpr']);r['delta_p01']=r['tpr']-p01[r['team']]
        r['tp_equivalent']=r['tpr']*r['members']
        r['source_ids']=';'.join(sourceids[s] for s in r.get('sources',[r['source']]))
        r['change_from_p01']=change_from_p01(r)
        bycfg[k].append(r)
    configs=[]
    for key in cfgkeys:
        rr=bycfg[key];first=rr[0]
        configs.append(dict(condition_id=cfgids[key],method=key[0],condition=key[1],evaluation_class=key[2],selection=key[3],
            attributes=first['attributes'],params=first['params'],background=first['background'],split=first['split'],transductive=first['transductive'],
            change_from_p01=first['change_from_p01'],source_method_aliases=';'.join(sorted({a for r in rr for a in r.get('aliases',[r.get('original_method',r['method'])])})),
            family=first['family'],source_ids=';'.join(sorted({s for r in rr for s in r['source_ids'].split(';')})),
            verification=';'.join(sorted({r['verification'] for r in rr})),notes=first['notes'],
            evaluated_teams=','.join(str(t) for t in sorted({r['team'] for r in rr})),team_count=len(rr),
            mean_tpr=sum(r['tpr'] for r in rr)/len(rr),band2_team_count=sum(r['band']>=2 for r in rr),
            band_up_from_p01=sum(r['band']>band(p01[r['team']]) for r in rr),
            lower_than_p01=sum(r['delta_p01'] < -1e-12 for r in rr)))
    fields=['condition_id','team','tpr','band','delta_p01','members','tp_equivalent','evaluation_class','source_ids']
    write_csv(out/'全方式評価.csv',sorted(rows,key=lambda r:(r['condition_id'],r['team'])),fields+['method','condition'])
    write_csv(out/'方式条件一覧.csv',configs,list(configs[0]))
    bandrows=[r for r in rows if r['band']>=2]
    write_csv(out/'帯2以上_全方式.csv',sorted(bandrows,key=lambda r:(r['team'],-r['tpr'],r['method'],r['condition'])),
        fields+['method','condition','attributes','params','background','change_from_p01','split','transductive','selection','verification','notes'])
    best=[]
    for team in sorted(p01):
        teamrows=[r for r in rows if r['team']==team]
        maximum=max(r['tpr'] for r in teamrows)
        ties=[r for r in teamrows if r['tpr']==maximum]
        winner=sorted(ties,key=lambda r:(r['evaluation_class']=='A',r['selection']!='fixed_rule',r['method'],r['condition']))[0]
        eligible=[r for r in teamrows if r['evaluation_class'] in ('B','C') and r['selection']=='fixed_rule']
        eligible_max=max((r['tpr'] for r in eligible),default=0)
        r=winner.copy();r.update(best_tie_count=len(ties),best_tie_condition_ids=';'.join(x['condition_id'] for x in ties),
            fixed_rule_BC_max=eligible_max,selection_for_report='公開正解による事後oracle（本戦で事前選択可能とは言えない）',
            proximity='帯2以上' if maximum>.25 else '>0.20' if maximum>.20 else '0.10〜0.20' if maximum>.10 else '<=0.10',
            p01_tpr=p01[team],p01_band=band(p01[team]),number_evaluations=len(teamrows))
        best.append(r)
    write_csv(out/'チーム別最高結果.csv',best,fields+['method','condition','attributes','params','background','split','selection','verification',
        'p01_tpr','p01_band','fixed_rule_BC_max','best_tie_count','best_tie_condition_ids','proximity','number_evaluations','selection_for_report'])
    sources=[]
    for s in sourcepaths:
        path=root/s.split('#',1)[0]
        sources.append(dict(source_id=sourceids[s],path=s,exists=path.exists(),sha256=hash_file(path) if path.is_file() else '',
                            bytes=path.stat().st_size if path.is_file() else ''))
    write_csv(out/'結果保存元.csv',sources,list(sources[0]))
    methods={r['method'] for r in rows}
    summary=dict(created_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),python=platform.python_version(),executable=sys.executable,
        raw_target_rows=raw_count,after_team12_exclusion_before_dedup=report_count,target_rows=len(rows),deduplicated_rows=report_count-len(rows),
        parameterized_methods=len(methods),method_conditions=len(configs),source_files=len(sources),
        report_teams=sorted(p01),band2_teams=sorted({r['team'] for r in bandrows}),
        new_band2_teams=sorted({r['team'] for r in bandrows if r['team'] not in (1,2)}),band2_rows=len(bandrows),
        evaluation_class_counts={c:sum(r['evaluation_class']==c for r in rows) for c in ('A','B','C')},
        excluded_report_teams=[5,12],fpr=.01,conflicts=conflicts,notes=notes,
        best=[{k:r[k] for k in ('team','tpr','method','condition','p01_tpr','band','proximity','evaluation_class')} for r in best],
        writes_submission_files=False,uploads_or_submits=False,
        code_sha256={f.name:hash_file(f) for f in Path(__file__).parent.glob('*.py')})
    (out/'集計記録.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    for name in ['全方式評価.csv','方式条件一覧.csv','帯2以上_全方式.csv','チーム別最高結果.csv','結果保存元.csv','集計記録.json']:
        if (args.output_dir/name).exists():raise FileExistsError('既存成果物を上書きしません: '+name)
    result=exports(args.root.resolve(),args.output_dir.resolve())
    print(json.dumps({k:v for k,v in result.items() if k not in ('notes','best','conflicts')},ensure_ascii=False))
    print(json.dumps({'best':result['best'],'conflicts':len(result['conflicts'])},ensure_ascii=False))

if __name__=='__main__':main()
