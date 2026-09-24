"""既存結果の再集計と共有ファイルの検証。実験・ネット通信・ファイル更新は行わない。"""
from pathlib import Path
import hashlib, json, math, statistics

HERE = Path(__file__).resolve().parent

def main():
    manifest = json.loads((HERE/'ファイル一覧.json').read_text())
    for item in manifest:
        p = HERE/item['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], item['file']
    d = json.loads((HERE/'集計用データ/追加比較.json').read_text())
    nmap = {p['id']:p['n'] for p in d['panels']}
    assert len(d['aia']) == 3360 and len(d['mia']) == 560
    assert len({(r['case'],r['panel'],r['attack']) for r in d['aia']}) == 3360
    for case in d['cases']:
        for n in [387,163]:
            vals=[]
            for pid in [p for p in nmap if nmap[p]==n]:
                scores=[r['AIA'] for r in d['aia'] if r['case']==case['id'] and r['panel']==pid]
                assert len(scores)==12
                vals.append(max(scores))
            summary=next(r for r in d['summary'] if r['case']==case['id'] and r['group']=='new' and r['target_n']==n)
            for key,value in [('mean',statistics.mean(vals)),('median',statistics.median(vals)),('minimum',min(vals)),('maximum',max(vals))]:
                assert math.isclose(value,summary[key],abs_tol=1e-12), (case['id'],n,key)
            if case['id'] in ['C27','C26']:
                print(case['name'],n,'AIA mean',f'{statistics.mean(vals):.6f}','max',f'{max(vals):.6f}')
        mrows=[r for r in d['mia'] if r['case']==case['id']]
        pids={r['panel'] for r in mrows}
        assert len(pids)==20
        vals=[]
        for pid in pids:
            scores=[r['TPR_at_FPR01'] for r in mrows if r['panel']==pid]
            assert len(scores)==4
            vals.append(max(scores))
        summary=next(r for r in d['mia_summary'] if r['case']==case['id'] and r['group']=='new')
        assert math.isclose(statistics.mean(vals),summary['mean'],abs_tol=1e-12)
        assert math.isclose(max(vals),summary['maximum'],abs_tol=1e-12)
    for row in json.loads((HERE/'Codabench公式結果.json').read_text()):
        raw=(HERE/'提出済み2案'/(row['name']+'.csv')).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row['C_sha256']
        assert math.isclose(row['official_U'],0.9219987812309569,abs_tol=1e-12)
    print('OK: files=',len(manifest),' AIA=3360, MIA=560, submitted C=2')

if __name__=='__main__':
    main()
