"""保存済み実験の実ファイルSHA照合と、任意で全背景の公式ROC再計算。

新しい方式選択や学習を行わない。過去の最大値を候補として推さない。
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import 共通 as u


def eligible(path):
    return not any(s=='失敗記録' or s.startswith(('再集計_','再現監査_')) for s in path.relative_to(u.HERE).parts)


def code_inventory():
    inventory={}
    for directory in [u.HERE,u.OLD,u.CAL]:
        for path in directory.rglob('*.py'):
            if '失敗記録' in path.parts:continue
            inventory.setdefault(u.sha(path),[]).append(str(path.resolve()))
    return inventory


def verify_hashes():
    inventory=code_inventory();rows=[]
    def check(record,kind,path,expected):
        if 'アーカイブ' in path.parts:raise PermissionError(path)
        actual=u.sha(path) if path.is_file() else None
        rows.append(dict(record=str(record.relative_to(u.HERE)),kind=kind,file=str(path),expected=expected,actual=actual,matches=actual==expected))
    for record in sorted(u.HERE.rglob('実行記録.json')):
        if not eligible(record):continue
        value=json.loads(record.read_text())
        for kind in ['input_sha256','source_sha256','support_code_sha256','output_sha256']:
            entries=value.get(kind,{})
            if not isinstance(entries,dict):continue
            for name,expected in entries.items():
                path=record.parent/name if kind=='output_sha256' else Path(name)
                check(record,kind,path,expected)
        if 'code_sha256' in value:
            expected=value['code_sha256']
            if value.get('code_path'):
                check(record,'executed_code_path',Path(value['code_path']),expected)
            else:
                candidates=inventory.get(expected,[])
                rows.append(dict(record=str(record.relative_to(u.HERE)),kind='actual_code_content',file=';'.join(candidates),
                                 expected=expected,actual=expected if candidates else None,matches=bool(candidates)))
        if 'feature_cache_sha256' in value:
            # 当該実行器の命名規約で参照元を一意に復元する。
            if record.parent.parent.name!='特徴除去の背景結果':raise ValueError(record)
            path=u.HERE/'背景変更の学習結果'/record.parent.name/'特徴と順位.npz'
            check(record,'feature_cache_sha256',path,value['feature_cache_sha256'])
    return pd.DataFrame(rows)


def aggregate_tables():
    rows=[];inputs={}
    for path in sorted(u.HERE.rglob('対象別TPR.csv')):
        if not eligible(path) or not (path.parent/'実行記録.json').exists():continue
        inputs[str(path)]=u.sha(path);table=pd.read_csv(path)
        for (method,fpr),part in table.groupby(['method','fpr']):
            assert len(part)==20 and part.team.nunique()==20,path
            scores=part.set_index('team').mia_all
            rows.append(dict(source=str(path.parent.relative_to(u.HERE)),method=method,fpr=float(fpr),all20=float(scores.mean()),other18=float(scores.drop([1,2]).mean())))
    return pd.DataFrame(rows),inputs


def verify_ranks(all_backgrounds=False):
    data=u.setup();rows=[];inputs={};models_total=0
    paths=[p for p in sorted(u.HERE.rglob('内部順位キャッシュ.npz')) if eligible(p) and (p.parent/'対象別TPR.csv').exists()]
    if not all_backgrounds:paths=[p for p in paths if len(p.relative_to(u.HERE).parts)<=2 or p.parent.name in ('full','採点')]
    for index,path in enumerate(paths,1):
        inputs[str(path)]=u.sha(path);z=np.load(path)
        assert list(z['teams'])==data['teams'],path
        if 'record_ids' in z:assert np.array_equal(z['record_ids'],data['b'].loc[data['normal'],'record_id'].to_numpy()),path
        models={str(name):matrix for name,matrix in zip(z['names'],z['scores'])}
        table,error,_=u.evaluate(data,models)
        saved=pd.read_csv(path.parent/'対象別TPR.csv')
        merged=table.merge(saved,on=['method','team','fpr'],suffixes=('_new','_saved'),validate='one_to_one')
        maximum=float(np.max(np.abs(merged.mia_all_new-merged.mia_all_saved)))
        assert maximum<1e-12 and len(merged)==len(table)==len(saved),path
        rows.append(dict(source=str(path.parent.relative_to(u.HERE)),models=len(models),score_values=len(table),saved_value_max_error=maximum,official_tpr_max_error=error))
        models_total+=len(models)
        if index%10==0 or index==len(paths):u.log(stage='audit_saved_predictions',completed=index,caches=len(paths),models=models_total,maximum_error=maximum)
    return pd.DataFrame(rows),inputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--verify-ranks',action='store_true')
    parser.add_argument('--all-backgrounds',action='store_true')
    args=parser.parse_args();started=time.perf_counter();now=datetime.now(ZoneInfo('Asia/Tokyo'))
    out=args.output_dir or u.HERE/('再現監査_'+now.strftime('%Y%m%d_%H%M%S'))
    assert not out.exists();out.mkdir()
    hashes=verify_hashes();hashes.to_csv(out/'実ファイルのハッシュ照合.csv',index=False)
    assert hashes.matches.all(),hashes[~hashes.matches].to_dict('records')
    table,inputs=aggregate_tables();table.to_csv(out/'18対象の保存結果.csv',index=False,float_format='%.17g')
    rank_checks=pd.DataFrame()
    if args.verify_ranks:
        rank_checks,rank_inputs=verify_ranks(args.all_backgrounds);inputs.update(rank_inputs)
        rank_checks.to_csv(out/'内部順位からの独立再採点.csv',index=False,float_format='%.17g')
    for path,expected in inputs.items():assert u.sha(path)==expected,path
    record=dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),seconds=time.perf_counter()-started,
                code_path=str(Path(__file__).resolve()),code_sha256=u.sha(__file__),input_sha256=inputs,
                hash_checks=len(hashes),hashes_match=bool(hashes.matches.all()),aggregated_settings=len(table),
                rank_cache_count=len(rank_checks),rank_model_count=int(rank_checks.models.sum()) if len(rank_checks) else 0,
                saved_value_max_error=float(rank_checks.saved_value_max_error.max()) if len(rank_checks) else None,
                official_tpr_max_error=float(rank_checks.official_tpr_max_error.max()) if len(rank_checks) else None,
                main_membership_labels_used=False,new_model_selection=False,all_backgrounds=args.all_backgrounds,
                code_hash_meaning='code_pathがある記録はその実ファイルを照合。ない記録も実ソースの内容が記録SHAと一致することを確認。実行場所は推定しない。',
                output_sha256={p.name:u.sha(p) for p in out.iterdir()})
    (out/'実行記録.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    u.log(stage='audit_complete',output=str(out),hash_checks=len(hashes),caches=record['rank_cache_count'],models=record['rank_model_count'],seconds=record['seconds'])


if __name__=='__main__':main()
