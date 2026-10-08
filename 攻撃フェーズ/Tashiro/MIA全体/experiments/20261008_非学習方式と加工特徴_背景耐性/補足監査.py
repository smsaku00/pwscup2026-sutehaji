"""集約表と、共通採点・既存full照合を支える素材の補足ハッシュ。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import background_common as bc

HERE=Path(__file__).resolve().parent

def main():
    records=[json.loads((HERE/(n+'_run.json')).read_text()) for n in ['geometry','hist_local','gen','legacy']]
    for record in records:
        for p,h in record['input_sha256'].items():assert bc.sha(p)==h,p
        for p,h in record['output_sha256'].items():assert bc.sha(HERE/p)==h,p
        for p,h in record['source_code_sha256'].items():assert bc.sha(bc.ROOT/p)==h,p
        assert record['members_used_only_after_all_predictions'] and not record['training']
        assert len(record['evaluation_teams'])==20 and 12 in record['evaluation_teams']
        assert len(record['report_teams'])==19 and 12 not in record['report_teams']
        assert max(record['full20_max_error_by_method'].values())<1e-12
    table=pd.read_csv(HERE/'固定方式_全背景評価.csv');stats=pd.read_csv(HERE/'背景耐性_全対象全方式.csv')
    assert len(table)==8778 and len(stats)==418
    assert not table.duplicated(['method','condition','team','fpr']).any()
    assert np.isfinite(table.tpr).all() and table.tpr.between(0,1).all()
    assert not table.team.isin([5,12]).any()
    for _,g in table.groupby(['method','condition','team']):
        assert (np.diff(g.sort_values('fpr').tpr)>=-1e-12).all()
    expected=bc.audit.load_official(bc.ROOT)
    distributed=bc.audit.child(bc.ROOT/'予備戦','配布データ')
    truthdir=bc.audit.child(distributed/'結果','正解データ')
    paths=set(expected._source_paths+[expected._source_paths[0].parent/'result.py',bc.PREVIOUS/'official_audit.py',Path(__file__)])
    paths.update(truthdir/f'pool_ids_{t}.csv' for t in sorted(set(records[0]['evaluation_teams'])|{5}))
    configs=pd.read_csv(bc.PREVIOUS/'方式条件一覧.csv');sources=pd.read_csv(bc.PREVIOUS/'結果保存元.csv').set_index('source_id')
    paths.update(bc.PREVIOUS/p for p in ['方式条件一覧.csv','全方式評価.csv','結果保存元.csv'])
    paths.update(bc.PREVIOUS.glob('historical*_cache.json'))
    names=set(table.method);used=configs[(configs.method.isin(names))&(configs.evaluation_class=='C')]
    for identifiers in used.source_ids:
        for sourceid in identifiers.split(';'):
            path=bc.ROOT/str(sources.loc[sourceid,'path']).split('#',1)[0]
            if path.suffix=='.csv' and path.exists():paths.add(path)
    record=dict(status='PASS',official_source_sha256=expected._source_sha256,
        supplemental_sha256={str(p.relative_to(bc.ROOT)):bc.sha(p) for p in sorted(paths)},
        hash_timing='実験完了後の補足監査。元の個別runの入力hashと併用。',
        rows=8778,methods=22,conditions=7,reported_teams=19,full_validation_teams=20,
        maximum_full_reproduction_error=max(max(r['full20_max_error_by_method'].values()) for r in records),
        label_read_rule='所属は各系列の全7背景の全予測が確定した後の採点で初読。',
        independent_qa='別agentが8778行の連結、418集約、26固定候補、公式再現、strict閾値を読み取り再集計して一致。',
        sensitive_rows_saved=False,competition_submission_created=False)
    path=HERE/'補足監査.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='supplemental_sha256'},ensure_ascii=False))

if __name__=='__main__':main()
