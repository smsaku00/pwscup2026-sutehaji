"""420件の的中人数・各手法表・統計・定数比較を再計算する。読み取り専用。"""
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
import csv
import math

HERE=Path(__file__).resolve().parent
TOL=1e-12


def read(name):
    with (HERE/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def same(a,b):assert math.isclose(float(a),float(b),abs_tol=TOL,rel_tol=TOL),(a,b)
def stats(v):return dict(zip(['最小','中央値','平均','最大'],[min(v),median(v),mean(v),max(v)]))
def score(hm,hc,nm,nc):return max(0,(hm/nm-hc/nc)/(1-hc/nc)) if hc<nc else 0


def main():
    rows=read('420件の詳細.csv');constants={int(r['パネル']):r for r in read('定数10年_パネル別.csv')}
    names=list(dict.fromkeys(r['加工案'] for r in rows));assert len(names)==7
    assert len(rows)==420 and len({(r['加工案'],r['攻撃手法'],r['パネル']) for r in rows})==420
    assert set(constants)==set(range(1,6))
    for r in constants.values():
        assert int(r['予測年数'])==10
        same(r['AIA得点'],score(*[int(r[k]) for k in ['標的的中人数','対照的中人数','標的人数','対照人数']]))
    groups,panels=defaultdict(list),defaultdict(list)
    for r in rows:
        p=int(r['パネル']);same(r['AIA得点'],score(*[int(r[k]) for k in ['標的的中人数','対照的中人数','標的人数','対照人数']]))
        same(r['定数10年の得点'],constants[p]['AIA得点']);same(r['定数との差'],float(r['AIA得点'])-float(constants[p]['AIA得点']))
        groups[r['加工案'],r['攻撃手法']].append(r);panels[r['加工案'],p].append(r)
    assert len(groups)==84 and all(len(g)==5 for g in groups.values())
    for r in read('手法別統計と定数比較.csv'):
        chosen=groups[r['加工案'],r['攻撃手法']]
        for field,prefix in [('AIA得点',''),('定数との差','定数との差_')]:
            for k,v in stats([float(x[field]) for x in chosen]).items():same(v,r[prefix+k])
        ds=[float(x['定数との差']) for x in chosen]
        for k,v in [('定数超え',sum(d>TOL for d in ds)),('定数と同じ',sum(abs(d)<=TOL for d in ds)),('定数未満',sum(d<-TOL for d in ds))]:same(v,r[k])
    for n in names:
        matrix=read(n+'_手法別パネル表.csv');assert len(matrix)==13
        for r in matrix:
            vs=[float(r['H'+str(i)]) for i in range(1,6)]
            expected=[float(constants[i]['AIA得点']) for i in range(1,6)] if r['区分']=='定数' else [float(x['AIA得点']) for x in sorted(groups[n,r['攻撃手法']],key=lambda x:int(x['パネル']))]
            for a,b in zip(vs,expected):same(a,b)
            for k,v in stats(vs).items():same(v,r[k])
    for r in read('パネル別比較.csv'):
        chosen=panels[r['加工案'],int(r['パネル'])];assert len(chosen)==12
        for k,v in stats([float(x['AIA得点']) for x in chosen]).items():same(v,r['12攻撃の'+k])
        mx=float(r['12攻撃の最大']);b=float(constants[int(r['パネル'])]['AIA得点'])
        same(r['定数10年'],b);same(r['最大と定数の差'],mx-b);same(r['定数込み13手法の最大'],max(mx,b))
    utilities={r['加工案']:r for r in read('評価したCと有用性.csv')}
    for r in utilities.values():same(r['U'],min(float(r[k]) for k in ['U_gen','U_spec','U_rare','U_valid']))
    for r in read('7案の要約.csv'):
        n=r['加工案'];chosen=[x for x in rows if x['加工案']==n];assert len(chosen)==60
        maxima=[max(float(x['AIA得点']) for x in panels[n,p]) for p in range(1,6)]
        included=[max(v,float(constants[p]['AIA得点'])) for p,v in enumerate(maxima,1)]
        for prefix,vs in [('全60件_',[float(x['AIA得点']) for x in chosen]),('パネル最大5値_',maxima),('定数込みパネル最大5値_',included)]:
            for k,v in stats(vs).items():same(v,r[prefix+k])
        ds=[float(x['定数との差']) for x in chosen]
        for k,v in [('定数超え件数',sum(d>TOL for d in ds)),('定数と同じ件数',sum(abs(d)<=TOL for d in ds)),('定数未満件数',sum(d<-TOL for d in ds))]:same(v,r[k])
        same(r['定数超えパネル数'],sum(v>float(constants[p]['AIA得点'])+TOL for p,v in enumerate(maxima,1)))
        same(r['U'],utilities[n]['U'])
    print('420件・7案の各手法×5パネル・最小/中央値/平均/最大・定数比較・U内訳を照合しました。')


if __name__=='__main__':main()
