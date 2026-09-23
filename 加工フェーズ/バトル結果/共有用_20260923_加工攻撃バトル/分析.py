"""同梱CSVを再集計・照合する。標準ライブラリのみ。ファイルの更新やバトル再実行はしない。"""
from collections import defaultdict
import csv,hashlib,json,math
from pathlib import Path
from statistics import median,mean

HERE=Path(__file__).resolve().parent
def rows(name):
    with (HERE/'summary'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def same(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-12,rel_tol=1e-12),(a,b)

def main():
    manifest=json.loads((HERE/'ファイル一覧とハッシュ.json').read_text())
    for r in manifest['files']:
        assert hashlib.sha256((HERE/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
    aia=rows('AIA全2085比較.csv');mia=rows('MIA全76比較.csv');panels=rows('AIAパネル別攻撃最大.csv')
    assert len(aia)==2085 and len(mia)==76 and len(panels)==170
    assert len({(r['suite'],r['panel'],r['defense'],r['attack']) for r in aia})==2085
    assert len({(r['suite'],r['defense'],r['attack']) for r in mia})==76
    assert {s:sum(r['suite']==s for r in aia) for s in ['main10','fixed5','main_MIA']}=={'main10':1800,'fixed5':240,'main_MIA':45}
    grouped=defaultdict(list)
    for r in aia:
        pm=int(r['member_hits'])/int(r['n_member']);pc=int(r['control_hits'])/int(r['n_control'])
        same(max(0,(pm-pc)/(1-pc)) if pc<1 else 0,r['R_all'])
        grouped[r['suite'],r['panel'],r['defense']].append(r)
    for r in panels:
        values=grouped[r['suite'],r['panel'],r['defense']]
        assert len(values)==12
        for name in ['old7','new4','reference','repo11','all12']:
            chosen=[v for v in values if name=='all12' or (name=='repo11' and v['group']!='reference') or v['group']==name]
            assert len(chosen)=={'old7':7,'new4':4,'reference':1,'repo11':11,'all12':12}[name]
            same(max(float(v['R_all']) for v in chosen),r[name+'_max'])
    full={r['defense']:r for r in rows('本戦全入力Cの有用性.csv')};assert len(full)==15
    for r in full.values():same(min(float(r[k]) for k in ['U_gen','U_spec','U_rare','U_valid']),r['U'])
    for r in rows('加工15案の比較.csv'):
        name=r['加工案'];same(full[name]['U'],r['本戦全入力CのU'])
        selected=[p for p in panels if p['suite']=='main10' and p['defense']==name];assert len(selected)==10
        for label in ['old7','new4','repo11','reference','all12']:
            same(max(float(p[label+'_max']) for p in selected),r[label+'_10パネル最大'])
        same(median(float(p['all12_max']) for p in selected),r['all12_パネル最大の中央値'])
        ms=[float(m['TPR_at_FPR001']) for m in mia if m['suite']=='main_MIA' and m['defense']==name];assert len(ms)==4
        same(max(ms),r['模擬MIA_4攻撃最大']);same(mean(ms),r['模擬MIA_4攻撃平均'])
        same(float(r['repo11_10パネル最大'])-float(r['old7_10パネル最大']),r['追加4攻撃による最大R増分'])
        same(float(r['all12_10パネル最大'])-float(r['repo11_10パネル最大']),r['公式参照による最大R増分'])
    utilities=rows('パネル生成Cの有用性.csv')
    for r in rows('固定C4案の比較.csv'):
        name=r['固定C'];selected=[p for p in panels if p['suite']=='fixed5' and p['defense']==name];assert len(selected)==5
        for label in ['old7','new4','repo11','reference','all12']:same(max(float(p[label+'_max']) for p in selected),r[label+'最大'])
        us=[p for p in utilities if p['suite']=='fixed5' and p['加工案名']==name];assert len(us)==5
        for u in us:same(u['U'],r['U'])
        ms=[float(m['TPR_at_FPR001']) for m in mia if m['suite']=='fixed5' and m['defense']==name];assert len(ms)==4
        same(max(ms),r['模擬MIA_4攻撃最大'])
    old=rows('0921同条件980比較.csv');assert len(old)==980
    for r in old:same(r['R_0921'],r['R_0923']);same(r['difference'],0)
    pair=rows('zhang04対11_120比較.csv');assert len(pair)==120
    for r in pair:same(float(r['zhang_11_normal_wald'])-float(r['zhang_04_copula_bgtime']),r['差_11引く04'])
    print(f"全{len(manifest['files'])}ファイルのハッシュ一致。AIA 2085件・MIA 76件・集約・U・旧980比較・ペア120比較を照合しました。")

if __name__=='__main__':main()
