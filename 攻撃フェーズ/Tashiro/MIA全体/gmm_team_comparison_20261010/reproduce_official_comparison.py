from pathlib import Path
import csv, json, hashlib, math, statistics, zipfile
import xml.etree.ElementTree as ET

import argparse
parser = argparse.ArgumentParser(description='Join fixed GMM aggregates to published preliminary scores.')
parser.add_argument('--workbook',type=Path,required=True)
parser.add_argument('--gmm-source',type=Path,required=True)
parser.add_argument('--utility-ledger',type=Path,required=True)
parser.add_argument('--out-dir',type=Path,required=True)
args=parser.parse_args()
OUT=args.out_dir
OUT.mkdir(parents=True,exist_ok=True)
XLSX=args.workbook
GMM=args.gmm_source
UTILITY=args.utility_ledger
METHOD = 'q13_robust_sex0_diag_reg0.01_gmm4__raw__cross0'
TEAMS = [1,2,3,4,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]
BGS = ['starter','prelim','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']
LABELS = ['starter','prelim','main','SP','SM','PM','all3']
NS = {'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
source_hash = hashlib.sha256(XLSX.read_bytes()).hexdigest()
assert source_hash == '8d6380feafa50a850b28aa50eb4b04cd0ffe132ca15904dc0c84b85fd8ff6508'
with zipfile.ZipFile(XLSX) as z:
    ss=[]
    if 'xl/sharedStrings.xml' in z.namelist():
        ss=[''.join(t.text or '' for t in si.iter('{'+NS['s']+'}t')) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',NS)]
    def sheet(number):
        data=[]
        for row in ET.fromstring(z.read(f'xl/worksheets/sheet{number}.xml')).findall('.//s:sheetData/s:row',NS):
            d={'row':int(row.get('r'))}
            for c in row.findall('s:c',NS):
                assert c.find('s:f',NS) is None
                v=c.find('s:v',NS)
                v=v.text if v is not None else None
                if c.get('t')=='s': v=ss[int(v)]
                if c.get('t')=='inlineStr': v=''.join(t.text or '' for t in c.iter('{'+NS['s']+'}t'))
                d[c.get('r').rstrip('0123456789')]=v
            data.append(d)
        return data
    summary=sheet(1)
    mia=sheet(4)
    aia=sheet(3)
    meta=sheet(7)

by_team={int(r['A']):r for r in summary[1:]}
attack5={int(r['B']):r for r in mia[1:] if r['A']=='5'}
aia_by_team={int(r['A']):r for r in aia[1:]}
old={int(r['team']):r for r in csv.DictReader(UTILITY.open(encoding='utf-8-sig'))}
gmm_rows=[]
with GMM.open(encoding='utf-8-sig',newline='') as f:
    for line,r in enumerate(csv.DictReader(f),start=2):
        if r['method']==METHOD:
            r['source_line']=line
            gmm_rows.append(r)
assert len(gmm_rows)==140
gmm={(int(r['team']),r['background']):r for r in gmm_rows}
assert len(gmm)==140
assert set(gmm)=={(t,b) for t in TEAMS for b in BGS}

def same(a,b):
    assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12),(a,b)

rows=[]
records=[]
for team in TEAMS:
    s=by_team[team]; a=attack5[team]; r=s['row']; ar=a['row']
    for c in 'CDEFGHIJKLMN': assert s.get(c) is not None
    same(s['C'],min(float(s[c]) for c in 'DEFG'))
    same(s['J'],max(.01,float(s['H'])-float(s['I'])+.01))
    same(s['K'],min(1,max(0,(float(s['J'])/.01-1)/11)))
    same(s['M'],1-max(float(s['K']),float(s['L'])))
    same(s['N'],min(float(s['C']),float(s['M'])))
    attacks=[x for x in mia[1:] if int(x['B'])==team]
    assert len(attacks)==20
    same(s['H'],statistics.mean(float(x['G']) for x in attacks))
    same(s['L'],max(float(v) for k,v in aia_by_team[team].items() if k not in ['A','row'] and v is not None))
    for field,col in [('U','C'),('U_gen','D'),('U_spec','E'),('U_rare','F'),('U_valid','G')]: same(s[col],old[team][field])
    assert s['B']==old[team]['team_name']
    assert a.get('E') is not None and a.get('G') is not None
    if int(a['D'])==0: assert a.get('F') is None
    else: assert a.get('F') is not None
    o={'チーム番号':team,'チーム名':s['B']}
    for bg,label in zip(BGS,LABELS): o[f'固定GMM_TPR全体_FPR0.01_{label}']=gmm[team,bg]['tpr_0.01']
    for field,col in [('U','C'),('U_gen','D'),('U_spec','E'),('U_rare','F'),('U_valid','G')]: o[f'公式有用性_{field}']=s[col]
    o.update({
        '公式MIA被害_正規化成分':s['K'],'公式匿名性_Anon':s['M'],
        '公式MIA_beta_raw_全攻撃者平均':s['H'],'公式MIA_beta_chance_偶然95パーセンタイル':s['I'],
        '公式MIA_beta_偶然補正後':s['J'],'公式AIA成分':s['L'],'公式総合得点_min_U_Anon':s['N'],
        '公式攻撃者5_TPR全体_FPR0.01':a['E'],'公式攻撃者5_TPR希少_FPR0.01':a.get('F',''),
        '公式攻撃者5_t_j_全体希少の悪い方':a['G'],'公式攻撃者5_標的メンバー数':a['C'],'公式攻撃者5_希少メンバー数':a['D'],
        '公式総括セル':f'総括!A{r}:N{r}',
        '公式有用性セル':f'総括!C{r}:G{r}（順にU,U_gen,U_spec,U_rare,U_valid）',
        '公式MIA被害セル':f'総括!K{r}','公式匿名性セル':f'総括!M{r}',
        '公式攻撃者5セル':f'攻撃_標的別_MIA!A{ar}:H{ar}（A=攻撃者,B=標的,E=全体TPR,F=希少TPR,G=t_j）',
        '公式結果URL':old[team]['source_url'],'公式結果ファイル':'公開用結果表_20260909.xlsx','公式結果SHA256':source_hash,
        '固定GMM方式':METHOD,'固定GMM集計CSV':'experiments/20261010_全方式背景7条件/promising_methods_0310UTC/fixed_methods_all_teams_seven_backgrounds.csv',
        '固定GMM出典行_背景順':' / '.join(f'{label}:{gmm[team,bg]["source_line"]}' for bg,label in zip(BGS,LABELS)),
        '固定GMM個別出典_背景順':' / '.join(f'{label}:{gmm[team,bg]["source_path"]}' for bg,label in zip(BGS,LABELS)),
        '背景略記':'SP=starter+prelim; SM=starter+main; PM=prelim+main; all3=starter+prelim+main',
        '得点の読み方':'U・U各成分・Anonは高いほど良い。MIA被害成分・AIA成分は高いほど被害が大きい。固定GMM列は今回の実験TPR、公式列は公表済み予備戦結果。GMMのTPRを公式被害成分とみなさない。',
        '公式集計定義':'t_j=max(全体TPR,希少TPR)、希少メンバーなしは全体のみ。beta_raw=20攻撃者のt_j平均。beta=max(beta_raw-beta_chance+0.01,0.01)。MIA成分=clip((beta/0.01-1)/11,0,1)。Anon=1-max(MIA成分,AIA成分)。',
        '欠損の扱い':'数値0は実測・掲載の0。公式希少TPRの空欄は希少メンバーなしによる非評価。対象20チームのGMM7背景・公式U4/U・MIA成分・Anon・攻撃者5全体TPRに欠損なし。',
        '仕様出典':'予備戦ルールブックprelim-attack-20260825 §6.1、§6.2、§6.2.1（pp.17–18）',
        '仕様URL':'https://github.com/pwscup/pwscup2026-kit/blob/4477277e18a8e3979ec6015838fb672744fb64b6/rulebook/PWS_Cup2026_ルールブック_prelim-attack-20260825.pdf'
    })
    rows.append(o)
    records.append({'team':team,'team_name':s['B'],'summary_row':r,'attack5_row':ar,'U':float(s['C']),'U_gen':float(s['D']),'U_spec':float(s['E']),'U_rare':float(s['F']),'U_valid':float(s['G']),'beta_raw':float(s['H']),'beta_chance':float(s['I']),'beta':float(s['J']),'official_mia_component':float(s['K']),'official_aia_component':float(s['L']),'Anon':float(s['M']),'official_attacker5_tpr_all':float(a['E']),'official_attacker5_t_j':float(a['G']),'gmm':{bg:float(gmm[team,bg]['tpr_0.01']) for bg in BGS}})

OUT.mkdir(parents=True,exist_ok=True)
dest=OUT/'予備戦20チーム_固定GMM7背景と公式有用性_MIA被害_匿名性.csv'
with dest.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with dest.open(encoding='utf-8-sig',newline='') as f:
    check=list(csv.DictReader(f))
assert len(check)==20 and list(map(lambda x:int(x['チーム番号']),check))==TEAMS
assert dest.read_bytes().startswith(b'\xef\xbb\xbf')
assert hashlib.sha256(XLSX.read_bytes()).hexdigest()==source_hash
report={'official_source_sha256':source_hash,'gmm_source_sha256':hashlib.sha256(GMM.read_bytes()).hexdigest(),'csv_rows':len(check),'csv_columns':len(rows[0]),'gmm_points':140,'required_missing_count':0,'official_mia_zero_teams':[r['team'] for r in records if r['official_mia_component']==0],'checks':['U=min(U_gen,U_spec,U_rare,U_valid)','published U4/U match prior extraction','beta=max(beta_raw-beta_chance+0.01,0.01)','MIA_component=clip((beta/0.01-1)/11,0,1)','Anon=1-max(MIA_component,AIA_component)','total=min(U,Anon)','beta_raw=mean of 20 attacker t_j values','AIA_component=max of per-attacker AIA values','20 teams x 7 unique GMM backgrounds','attacker5 rare TPR missing only for zero rare members','source workbook unchanged'],'records':records}
(OUT/'official_join_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(dest)
print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False,indent=2))
for q in records: print(f"{q['team']}: U={q['U']:.4f}, MIA={q['official_mia_component']:.4f}, Anon={q['Anon']:.4f}, 公式5TPR={q['official_attacker5_tpr_all']:.4f}")
