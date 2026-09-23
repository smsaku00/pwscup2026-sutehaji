"""同梱CSVだけで360評価・背景標本10000件・統計を再計算。ファイル変更なし。"""
from pathlib import Path
from statistics import mean,median
import csv,math
HERE=Path(__file__).resolve().parent
def read(n):
    with (HERE/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def same(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-12,rel_tol=1e-12),(a,b)
def score(m,c):return max(0,(m-c)/(163-c)) if c<163 else 0
def stats(v):return dict(zip(['最小','中央値','平均','最大'],[min(v),median(v),mean(v),max(v)]))
def main():
    detail=read('360件の詳細.csv');null=read('背景標本10000件.csv');base=read('パネル別基準.csv');assert len(detail)==360 and len(null)==10000 and len(base)==10
    assert len({(r['対照の出所'],r['加工案'],r['攻撃手法'],r['パネル']) for r in detail})==360
    sc={'prelim':'予備戦対照','practice':'練習対照'};bs={}
    for r in null:same(r['R'],score(int(r['member_hits']),int(r['control_hits'])))
    for r in base:
        vs=[float(x['R']) for x in null if x['scenario']==r['scenario'] and x['panel']==r['panel']];assert len(vs)==1000
        same(r['reference_mean'],mean(vs));same(r['reference_median'],median(vs));same(r['constant_R'],score(int(r['constant_member_hits']),int(r['constant_control_hits'])))
        bs[sc[r['scenario']],r['panel']]=r
    for r in detail:
        assert int(r['標的人数'])==int(r['対照人数'])==163
        same(r['AIA得点'],score(int(r['標的的中人数']),int(r['対照的中人数'])))
        b=bs[r['対照の出所'],r['パネル']];same(r['定数10年'],b['constant_R']);same(r['背景標本基準'],b['reference_mean']);same(r['定数との差'],float(r['AIA得点'])-float(r['定数10年']))
    for r in read('手法別統計.csv'):
        chosen=[x for x in detail if all(x[k]==r[k] for k in ['対照の出所','加工案','攻撃手法'])];assert len(chosen)==5
        for k,v in stats([float(x['AIA得点']) for x in chosen]).items():same(r[k],v)
        ds=[float(x['定数との差']) for x in chosen]
        for k,v in [('定数超え',sum(x>1e-12 for x in ds)),('定数と同じ',sum(abs(x)<=1e-12 for x in ds)),('定数未満',sum(x<-1e-12 for x in ds))]:same(r[k],v)
    panels=read('パネル別比較.csv')
    for r in panels:
        chosen=[x for x in detail if all(x[k]==r[k] for k in ['対照の出所','加工案','パネル'])];assert len(chosen)==12
        values=[float(x['AIA得点']) for x in chosen]
        for k,v in stats(values).items():same(r['12攻撃の'+k],v)
        b=bs[r['対照の出所'],r['パネル']];inc=max(max(values),float(b['constant_R']));adj=max(0,inc-float(b['reference_mean']))
        same(r['定数込み最大'],inc);same(r['模擬基準差引後'],adj);same(r['AIAだけの参考保護値'],1-adj)
    u={r['加工案']:r for r in read('評価したCと有用性.csv')}
    for r in u.values():same(r['U'],min(float(r[k]) for k in ['U_gen','U_spec','U_rare','U_valid']))
    for r in read('3案の要約.csv'):
        chosen=[x for x in panels if all(x[k]==r[k] for k in ['対照の出所','加工案'])];assert len(chosen)==5
        rows=[x for x in detail if all(x[k]==r[k] for k in ['対照の出所','加工案'])];assert len(rows)==60
        for prefix,values in [('パネル最大5値_',[float(x['12攻撃の最大']) for x in chosen]),('全60件_',[float(x['AIA得点']) for x in rows])]:
            for k,v in stats(values).items():same(r[prefix+k],v)
        same(r['U'],u[r['加工案']]['U']);same(r['定数込み最大'],max(float(x['定数込み最大']) for x in chosen))
        adj=[float(x['模擬基準差引後']) for x in chosen]
        same(r['模擬基準差引後中央値'],median(adj));same(r['模擬基準差引後最大'],max(adj));same(r['AIAだけの参考保護値最小'],1-max(adj))
        matrix=read(r['対照の出所']+'_'+r['加工案']+'_手法別パネル表.csv');assert len(matrix)==14
        for m in matrix:
            for i in range(1,6):
                if m['区分'] in ['定数','背景基準']:v=float(bs[r['対照の出所'],str(i)]['constant_R' if m['区分']=='定数' else 'reference_mean'])
                else:v=next(float(x['AIA得点']) for x in rows if x['攻撃手法']==m['攻撃手法'] and int(x['パネル'])==i)
                same(m['P'+str(i)],v)
            for k,v in stats([float(m['P'+str(i)]) for i in range(1,6)]).items():same(m[k],v)
    gaps=read('背景基準との差_要約.csv');assert len(gaps)==6
    assert len({(r['対照の出所'],r['加工案']) for r in gaps})==6
    for r in gaps:
        chosen=[x for x in panels if all(x[k]==r[k] for k in ['対照の出所','加工案'])];assert len(chosen)==5
        adj=[max(0,float(x['定数込み最大'])-float(x['背景標本基準'])) for x in chosen]
        worst=chosen[adj.index(max(adj))]
        for x in chosen:same(x['12攻撃の最大'],x['定数込み最大'])
        same(r['差引後最大のパネル'],worst['パネル'])
        same(r['そのパネルのAIA最大（定数込み）'],worst['定数込み最大'])
        same(r['同じパネルの背景基準'],worst['背景標本基準'])
        same(r['AIA最大−背景基準（0未満は0）'],max(adj))
        same(r['1−差引後（AIAだけの参考値）'],1-max(adj))
        for k,v in stats(adj).items():same(r['差引後5パネル_'+k],v)
    print('360評価・背景標本10000件・各統計・U内訳・同一パネルで引いた背景基準との差と最悪パネルを照合しました。')
if __name__=='__main__':main()
