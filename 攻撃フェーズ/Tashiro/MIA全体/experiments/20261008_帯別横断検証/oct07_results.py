"""10月7日と10月8日広範較正の保存対象別評価を読む。入力への書込みなし。"""
from pathlib import Path
import csv
import re

QI7 = 'age,sex,prefecture,SBP,TG,HDL,ALT'
QI9 = 'age,sex,prefecture,BMI,SBP,TG,HDL,ALT,FPG'

FILES = [
 ('結果/予備戦_対象別比較.csv','geometry'),
 ('追加結果_最終版/予備戦_追加対象別比較.csv','geometry'),
 ('参照結果_拡張版/参照_対象別比較.csv','reference'),
 ('背景再標本化結果/背景再標本化_対象別比較.csv','bootstrap'),
 ('学習結果_順位校正版/学習_対象別比較.csv','learning'),
 ('属性部分集合結果/部分集合_対象別比較.csv','subsets'),
 ('低FPR結果/低FPR_対象別比較.csv','fixed'),
 ('固定候補の感度結果/感度検証_対象別比較.csv','fixed'),
 ('非会員較正結果/非会員較正_対象別比較.csv','calibration'),
 ('非会員較正の感度結果/較正感度_対象別比較.csv','calibration'),
 ('非会員較正の感度結果/較正低FPR_対象別比較.csv','calibration'),
]

def normalize(method, family):
    condition = 'full'
    if '__' in method:
        condition, method = method.split('__',1)
    original = method
    if family == 'reference':
        hit = re.match(r'(full|half[01])_(bg(?:_cross(?:0\.1|0\.25))?|aug_.*)$',method)
        if hit:
            condition, kind = hit.groups()
            if kind=='bg':method='global7'
            elif kind.startswith('bg_cross'):method='global7_cross_loo_'+kind.split('cross',1)[1]
            else:method='reference_'+kind
        elif method in ('global7_cross0.1','global7_cross0.25'):
            method='global7_cross_loo_'+method.split('cross',1)[1]
    aliases = {'baseline':'global7','raw':'global7','sensitivity_baseline':'global7',
               'fixed_cross01':'global7_cross_loo_0.1','raw_cross01':'global7_cross_loo_0.1',
               'global7_cross01':'global7_cross_loo_0.1',
               'baseline_cross01':'global7_cross_loo_0.1',
               'baseline_cross0.1':'global7_cross_loo_0.1',
               'baseline_cross0.25':'global7_cross_loo_0.25',
               'global7_leave_target_out_cross025':'global7_cross_loo_0.25',
               'Q_pref_sex100':'Q_prefecture_sex_shrink100',
               'Q_pref_sex500':'Q_prefecture_sex_shrink500',
               'Q_pref_sex100_cross01':'Q_prefecture_sex_shrink100_cross01',
               'Q_pref_sex500_cross01':'Q_prefecture_sex_shrink500_cross01',
               'augmented':'reference_aug_plain',
               'augmented_cross0.1':'reference_aug_plain_cross0.1'}
    method = aliases.get(method,method)
    if original.startswith('leave_target_out_alpha_'):
        method = 'global7_cross_loo_'+original.rsplit('_',1)[1]
    elif original.startswith('cross_alpha_'):
        method = 'global7_cross_all_'+original.rsplit('_',1)[1]
    elif original.startswith('sensitivity_cross_'):
        suffix=original[len('sensitivity_cross_'):]
        method='global7_cross_all_'+suffix
    elif original=='global7_cross025':
        method='global7_cross_all_0.25'
    # 制御CDFは元の順序を保つことが保存コードで検証済み。
    if method in ('Q_global_control','A_loo_global_control'):
        method='global7'
    return method,condition

def metadata(method, family):
    attributes=QI9 if method.startswith(('global9','sex9')) else QI7
    cols=['age','SBP','TG','HDL','ALT']
    hit=re.match(r'drop_(age|SBP|TG|HDL|ALT)(?:_cross01)?$',method)
    if hit: attributes=','.join([x for x in cols if x!=hit[1]]+['sex','prefecture'])
    for sub in ('age_TG_HDL','SBP_TG_HDL','age_SBP_ALT'):
        if method.startswith(sub): attributes=sub.replace('_',',')+',sex,prefecture'
    if method.startswith(('drop_mean','drop_median','drop_max','baseline075')):
        attributes=QI7+'（各1属性除去の合成）'
    learning='learn_' in method
    attrs='logit_rank7;logit_other_median7' if method.endswith('nuisance2') else attributes
    if method.endswith('signals5'): attrs='rank7,other_median7,other_max7,sex_rank7,rank9（logit）'
    params=['元設定名='+method]
    if 'point_h' in method:
        hit=re.search(r'point_h([0-9.]+)',method)
        params.extend(['点KDE密度比（近傍k未使用）','bandwidth_multiplier='+hit[1],'rho=.98','TG/ALT=log1p','IQR/1.349'])
    elif 'nn_k' in method:
        hit=re.search(r'nn_k([0-9]+)',method)
        params.extend(['近傍距離比','k='+hit[1],'カテゴリ不一致罰則=1','TG/ALT=log1p','IQR/1.349'])
    elif method.startswith(('global','sex','drop','reference','Q_','A_')):
        hit=re.search(r'sex7_k([13])',method)
        params.extend(['Gen-k='+ (hit[1] if hit else '10'),'rho=.98','TG/ALT=log1p','IQR/1.349'])
    if family=='bootstrap': params.extend(['draws=20','seed=20261007','sex×prefecture層内bootstrap'])
    return dict(attributes=attrs,params=';'.join(params),
                evaluation_class='B' if learning else 'C',
                split='人物5分割×対象4分割、seed20261007、公開875候補でfold順位較正' if learning else '正解は採点にのみ使用（既知正解を用いた探索履歴あり）',
                transductive='公開C・B候補順位、crossでは他Cを参照',
                family='学習2/5特徴' if learning else family,
                notes='希少11行=.4、既知非会員163行=0、通常875行>.6。cross_allは自対象を中央値に含む。')

def gather(root):
    root=Path(root)
    base=root/'本戦/攻撃フェーズ/Tashiro/20261007_MIA全体改善の検討'
    records=[]
    notes=[]
    calbase=root/'本戦/攻撃フェーズ/Tashiro/20261008_MIA全体_較正の広範比較'
    with (calbase/'結果/背景条件.csv').open(encoding='utf-8-sig',newline='') as f:
        sizes={r['condition']:int(r['background_rows']) for r in csv.DictReader(f)}
    sizes.update(half0=2572,half1=2571)
    def background(method,cond):
        value='予備戦公開A（母集団5143行）;条件='+cond
        if cond in sizes:value+=';使用A行数='+str(sizes[cond])
        if method.startswith('reference_aug_') and cond in sizes:
            value+=';foreignQ追加6520行;参照総行数='+str(sizes[cond]+6520)
        return value
    for rel,family in FILES:
        path=base/rel
        if not path.exists():
            notes.append({'missing':str(path.relative_to(root))}); continue
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                if float(row.get('fpr','.01'))!=.01: continue
                method,cond=normalize(row['method'],family)
                meta=metadata(method,family)
                if family=='learning' and method.endswith('rank1'):
                    method='global7' # 1特徴単調変換の順位一致をコードと保存表で確認
                rec=dict(team=int(row['team']),method=method,original_method=row['method'],condition=cond,tpr=float(row['mia_all']),
                         members=int(row.get('members') or 0),source=str(path.relative_to(root)),
                         verification='保存対象別TPR・既存集計監査（全入力からの再実行は別記）',
                         background=background(method,cond))
                rec.update(meta)
                rec['params']+=';保存元設定名='+row['method']
                if method=='global7': rec['family']='Gen-LRA7'
                records.append(rec)
    notes.extend([
      {'excluded':'参照結果/・追加結果/は拡張版/最終版に同設定を含む重複。学習結果/はfold尺度未較正版の撤回結果、順位校正版を使用'},
      {'excluded':'補助結果/対照一致はAIA非共有者と対照のproxy、MIA全体ではない'},
      {'selection':'全設定の再利用評価。学習2/5特徴はB、非学習はCだが方式選択自体は公開正解を繰り返し使用'},
    ])
    for rel in ['結果/対象別TPR.csv','追加混合/結果/対象別TPR.csv']:
        path=calbase/rel
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                if float(row['fpr'])!=.01: continue
                name=row['method']; method=name
                if name=='raw':method='global7'
                elif name=='raw_cross01':method='global7_cross_loo_0.1'
                elif name in ('flat_pref_sex_s100','flat_pref_sex_s500'):
                    method='Q_prefecture_sex_shrink'+name.rsplit('s',1)[1]+'_cross01'
                elif name in ('flat_sex_s100','flat_sex_s500'):
                    method='Q_sex_shrink'+name.rsplit('s',1)[1]+'_cross01'
                records.append(dict(team=int(row['team']),method=method,original_method=name,condition=row['condition'],tpr=float(row['mia_all']),
                    members=int(row['members']),attributes=QI7,params='元設定名='+name+';Gen-k10;cross=0.1（raw以外）;年齢境界はfullAの20/40/60/80分位で固定',
                    background=background(method,row['condition']),evaluation_class='C',split='正解は採点だけ。65方式の結果を見た追加24混合、正解再利用探索',
                    transductive='対象C・foreignQ・他C順位',source=str(path.relative_to(root)),
                    verification='保存対象別TPR・公式照合済み既存記録',family='較正・階層・可変・混合' if method!='global7' else 'Gen-LRA7',
                    notes='Q県×性別/性別/性別×年齢、縮約0,25,50,100,200,300,500,750,1000,2000,5000。設定名に係数を含む。'))
    return records,notes
