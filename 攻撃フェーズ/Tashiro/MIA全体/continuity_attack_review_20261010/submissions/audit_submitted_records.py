from pathlib import Path
import csv,json,hashlib,zipfile,xml.etree.ElementTree as ET,re,html
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
PRIVATE=Path('/workspace/shared/mia_recovered_validation/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/private_inputs')
XLSX=PRIVATE/'公開用結果表_20260909.xlsx'
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(XLSX) as z:
 def sheet(n):
  rows=[]
  for row in ET.fromstring(z.read(f'xl/worksheets/sheet{n}.xml')).findall('.//s:sheetData/s:row',NS):
   item={'row':int(row.get('r'))}
   for c in row.findall('s:c',NS):
    assert c.find('s:f',NS) is None
    v=c.find('s:v',NS);v=v.text if v is not None else None
    if c.get('t')=='inlineStr':v=''.join(t.text or '' for t in c.iter('{'+NS['s']+'}t'))
    item[c.get('r').rstrip('0123456789')]=v
   rows.append(item)
  return rows
 attacks={int(r['B']):r for r in sheet(4)[1:] if r['A']=='5'}
 meta={r['A']:r['B'] for r in sheet(7)[1:]}
file8=PRIVATE/'prelim_submission8_F_mia.csv'
hash8=hashlib.sha256(file8.read_bytes()).hexdigest()
assert hash8==meta['sha256.attack.5.F_mia.csv']=='cecc02572e677666d9b42283e25a8a656801b804cd83b25afc799adc700fffa3'
a=pd.read_csv(file8,float_precision='round_trip')
s=json.loads((PRIVATE/'prelim_submission7_fixed_settings.json').read_text())
fr=Path('/workspace/shared/mia_all_bg_legacy/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/recovered_archive/followup/results/prelim/submission7')
m=json.loads((fr/'prediction_manifest.json').read_text())
assert hashlib.sha256((fr/'frozen.npz').read_bytes()).hexdigest()==m['freeze_sha256']
x=np.load(fr/'frozen.npz')['legacy_submission_7']
assert list(a.columns[1:])==[str(t) for t in m['teams']]
rare=np.isin(a.record_id,s['private_rare_ids']+s['eligible_ids'])
assert rare.sum()==24 and (~rare).sum()==1025
assert np.array_equal(a.iloc[:,1:].to_numpy()[~rare],x[~rare])
assert int((a.iloc[:,1:].to_numpy()!=x).sum())==7

snap=json.loads((OUT/'main_official_receipt_snapshots.json').read_text())
raw=[]
for r in snap['snapshots']:
 d=json.loads(html.unescape(re.search(r'<pre>(.*?)</pre>',r['content'],re.S).group(1)))
 assert d['constants']['low_fpr']==.01
 raw.append({'round':r['round'],'mean_tpr':d['mia']['tpr_at_low_fpr'],'per_target_band':d['mia_per_target'],'source_url':r['url']})
settings={
 15:('可変幅KDE密度比','q7_robust_adaptivek15_h1__age100__cross0.1','C側k15・BG側k74、幅倍率1、性別×年齢5分位の非会員較正100、他対象中央値補正0.1','保存生成コード・設定・内部NPZ・CSV照合済み。今回クラウド独立再生成なし'),
 5:('性別内中央値/IQR整列・最近傍k1','q7_median_iqr_sex1_dcr_k1__raw__cross0','性別内中央値/IQR整列、k1、性別・県不一致に各1、非会員較正なし、他対象補正なし','保存生成コード・設定・内部NPZ・CSV照合済み。今回クラウド独立再生成なし'),
 6:('Gen-LRA7・年齢較正','GenLRA7_age100','性別×年齢5分位の非会員較正100、他対象中央値補正0.1','通常875人を独立再現、float64全要素一致'),
}
p01=('Gen-LRA7・県性別較正','P01','県×性別の非会員較正500、他対象中央値補正0.1。第3回通常者回答を保持','通常875人を独立再現、float64全要素一致')
pairs=[('さいふぉん-3',20,15),('神神公社',7,5),('CyberMantra',8,1),('k等分の花嫁',12,6),('Trailblazer',6,7),('Ritz',1,12),('あかぎ団',9,13)]
round4_source='https://github.com/smsaku00/pwscup2026-sutehaji/blob/8d53b32093e1e05a8641a11a818c670c0d6e231b/攻撃フェーズ/Tashiro/MIA全体/rounds/20261010_第4回方式照合/README.md'
rows=[]
for name,p,k in pairs:
 ar=attacks[p];j=m['teams'].index(p)
 assert all(r['per_target_band'][str(k)]==1 for r in raw)
 method,methodid,params,validation=settings.get(k,p01)
 normal_method='第6回の通常者回答をそのまま保持（元の生成方式は未確認）' if p==1 else 'Gen-LRA QI7：age/SBP/TG/HDL/ALT＋sex/prefecture、h倍率1、k10、rho0.98、TG/ALT log1p、予備戦BGのIQR/1.349尺度'
 rows.append({
  'チーム名':name,'予備戦番号':p,'本戦番号':k,
  '予備戦_公式採用提出TPR全体':float(ar['E']),
  '予備戦_最高値確定状況':'全8回中の最高値は未確定。公式採用第8回の値',
  '予備戦_通常者方式':normal_method,
  '予備戦_方式照合':'第8回の非希少1025行は保存第7回生成コードの再現値と全セル完全一致',
  '予備戦_希少差分対第7回':int((a[str(p)].to_numpy()[rare]!=x[rare,j]).sum()),
  '予備戦_希少方式注記':'Gen-LRA 9属性＋既知非会員除外を基礎とし、第8回で一部変更。通常者方式と全体提出を区別',
  '予備戦_公式セル':f'攻撃_標的別_MIA!E{ar["row"]}',
  '予備戦_公式URL':'https://drive.google.com/file/d/1BLiwflR7FJ9tp5lCeSkP-kJga4ByHRnB/view',
  '本戦_最高TPR全体':None,'本戦_各回帯':'第1回=1; 第2回=1; 第3回=1; 第4回=1',
  '本戦_最高値確定状況':'正確な対象別TPR非公開。4回とも0以上0.25以下。最高回・最高方式は判定不能',
  '本戦_最新提出回':4,'本戦_最新提出ID':969661,
  '本戦_第4回通常者方式':method,'本戦_第4回方式ID':methodid,'本戦_第4回設定':params,
  '本戦_第4回方式証拠':validation,'本戦_第4回方式URL':round4_source
 })
out={'as_of_utc':'2026-10-10T10:56:00Z','score':'MIA全体TPR at FPR0.01','our_attacker_prelim':5,'our_attacker_main':28,
 'prelim_official_workbook_sha256':hashlib.sha256(XLSX.read_bytes()).hexdigest(),
 'prelim_selected_mia_sha256':hash8,
 'prelim_source_equivalence':{'round8_vs_reproduced_round7_nonrare_1025x20':'bitwise numeric identical','round8_vs_round7_differences':7,'all_differences_in_rare24':True,'reference_npz_sha256':m['freeze_sha256']},
 'prelim_history_limit':'公式採用第8回の提出CSV・公式結果表は確認済み。全8回の公式レシートと対象別履歴は今回取得済み資料にない。第6・7回の生成物/保存条件だけを提出済み最高値に数えない',
 'main_rounds':raw,'main_aggregate_best':{'round':3,'submission_id':967396,'mean_tpr':max(r['mean_tpr'] for r in raw),'method':'Gen-LRA7＋県×性別の非会員較正500＋他対象中央値補正0.1','note':'全22対象平均。7チーム各自の値ではない'},
 'main_post_round4':{'tree_sha_checked':'0c74d92485a4be264b64e61f49bd521a0ea427b8','tree_truncated':False,'post4_receipts_found':False,'AIA_round5_explicitly_unsubmitted':True,'Yamazaki_round5_explicitly_unsubmitted':True,'note':'AIA第5回案はMIA第3回をコピーした未提出候補。Yamazaki第5回MIA案も未提出。新しい公式実績として数えない'},
 'rows':rows}
(OUT/'submitted_records_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
with (OUT/'seven_team_submitted_records.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(json.dumps({'records':len(rows),'main_rounds_verified':len(raw),'aggregate_best':out['main_aggregate_best'],'nonrare_exact_match':True},ensure_ascii=False))

