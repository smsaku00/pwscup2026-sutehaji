"""本戦22の公開Q/C属性支持を双方向に数える固定必要条件診断。

事前指定12setsを維持し、QにないBMI/FPGは不可評価として空欄にする。
公開患者行やIDを出力せず、B/GT/攻撃score/順位/採点を使用しない。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
COUNTY=HERE/'公開Q県集合の行対応必要条件.json'
FAILED=HERE/'公開QとCの双方向支持必要条件.py'
FAILED_OUT=HERE/'公開QとCの双方向支持必要条件結果'
FIELDS=('age','sex','prefecture','BMI','SBP','TG','HDL','ALT','FPG')
SETS=(('age',('age',)),('sex',('sex',)),('prefecture',('prefecture',)),
    ('prefecture+sex',('prefecture','sex')),('sex+age',('sex','age')),('prefecture+sex+age',('prefecture','sex','age')),
    ('BMI',('BMI',)),('SBP',('SBP',)),('TG',('TG',)),('HDL',('HDL',)),('ALT',('ALT',)),('FPG',('FPG',)))
PRIVATE_N=163


def guard(event,args):
    if event!='open' or not args or isinstance(args[0],int):return
    path=unicodedata.normalize('NFC',str(args[0]))
    if any(s in path for s in ('B_self','pool_ids','/正解データ/','/アーカイブ/','.npz','F_mia','F_aia','attack_score.py','対象別TPR.csv','result.txt')):
        raise PermissionError('公開Q/C必要条件では参照禁止: '+path)


sys.addaudithook(guard)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()
def write_json(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def paths():
    folder=HERE.parent/'20260928_配布データ基本分析/入力'
    result=[(int(p.stem.split('_')[1]),p,folder/f'aia_challenge_{int(p.stem.split("_")[1])}.csv')
        for p in sorted(folder.glob('C_*.csv')) if int(p.stem.split('_')[1])!=28]
    assert len(result)==22 and all(q.is_file() for _,c,q in result)
    return result


def header(p):
    with p.open(newline='') as f:return next(csv.reader(f))


def observed(frame,cols):return Counter(frame[list(cols)].itertuples(index=False,name=None))


def directions(c,q,cols):
    cc,qc=observed(c,cols),observed(q,cols)
    common=set(cc)&set(qc)
    forward=sum(cc[key] for key in common);reverse=sum(qc[key] for key in common)
    cf=pd.MultiIndex.from_frame(c[list(cols)]).isin(pd.MultiIndex.from_frame(q[list(cols)]))
    qf=pd.MultiIndex.from_frame(q[list(cols)]).isin(pd.MultiIndex.from_frame(c[list(cols)]))
    assert forward==int(cf.sum()) and reverse==int(qf.sum())
    return forward,reverse,len(common),len(cc),len(qc)


def csv_counters(p,sets):
    # Independent scalar parser: no patient rows, tuples or identifiers are exported.
    out={name:Counter() for name,cols in sets}
    with p.open(newline='') as f:
        for row in csv.DictReader(f):
            for name,cols in sets:
                key=tuple(row[col] if col=='prefecture' else float(row[col]) for col in cols);assert all(np.isfinite(v) for v in key if not isinstance(v,str))
                out[name][key]+=1
    return out


def numerical_qa():
    c=pd.DataFrame({'x':[1.,1.,1.,3.],'y':[4.,4.,4.,5.]});q=pd.DataFrame({'x':[1.,1.,2.],'y':[4.,4.,5.]})
    assert directions(c,q,('x','y'))==(3,2,1,2,2)
    for cc,qq in ((c.iloc[:0],q),(c,q.iloc[:0]),(c.iloc[:0],q.iloc[:0])):
        assert directions(cc,qq,('x','y'))[:3]==(0,0,0)
    return dict(duplicate_C_and_Q_occurrences_preserved=True,asymmetric_direction_example={'forward':3,'reverse':2},
        empty_cases=3,Counter_MultiIndex_count_max_error=0)


def proof(county):
    definition=county['official_definition'];assert definition['private_targets_in_Q']==PRIVATE_N and definition['external_controls_in_Q']==163
    assert definition['Q_rows']==326 and definition['private_Q_targets_subset_of_target_B']
    return dict(kind='fixed_public_support_necessary_conditions_not_final_report',official_definition=definition,
        official_definition_source_sha256=county['source_sha256'],private_target_occurrences=PRIVATE_N,
        weak_H='All 163 private target occurrences in Q have their observed attribute tuple somewhere in C support; distinct C rows are not assumed.',
        weak_necessary_condition='n_Q_in_Csupport >= 163',
        weak_proof='The private targets are 163 Q occurrences. Under weak_H each lies in C support, so the count of all Q occurrences in C support is at least 163.',
        strong_H='All 163 private target occurrences in Q correspond to distinct C row occurrences preserving the same observed tuple.',
        strong_necessary_conditions=['n_Q_in_Csupport >= 163','n_C_in_Qsupport >= 163'],
        strong_forward_proof='Each of the 163 distinct corresponding C occurrences has a tuple from Q support. Thus at least 163 C occurrences lie in Q support.',
        univariate_column_permutation='If a C column is a value- and multiplicity-preserving permutation of the full target B column, private target occurrences can map to distinct C occurrences for that single attribute. Both conditions are necessary.',
        joint_column_permutation='Separate permutations of individual columns need not preserve joint tuples. No joint-H inference follows merely from separate univariate preservation.',
        private_individuals_or_labels_known=False,threshold_source='163 from official Q construction, no optimized threshold',
        C_row_correspondence_required_by_schema=False,pass_proves_H=False,
        equality='Exact typed numeric equality and exact prefecture string equality of observed public fields, parsed round_trip. No rounding or tolerance matching.',
        unavailable_fields='A missing Q/C attribute is not evaluated; count and fail flags are blank, not zero or pass.',
        alternatives_not_identified=['synthetic output','resampling or duplication','row replacement','attribute transformations or recoding','other failure of tuple coverage or distinct occurrence representation'],
        proof_does_not_identify=['private patient identity','processing algorithm','which attribute changed','attack suitability','MIA accuracy','method to select'],
        membership_truth_used=False,B_used=False,attack_score_used=False,method_selection=False,final_prose_created=False,external_action=False)


def run(out,plan):
    assert sha(__file__)==plan['code_sha256'] and sha(COUNTY)==plan['county_condition_sha256']
    for p,h in plan['failed_attempt_sha256'].items():assert sha(p)==h
    county=json.loads(COUNTY.read_text());old={r['target']:r['C_rows_with_prefecture_in_Q_support'] for r in county['records']}
    records=[];available_cases=unavailable_cases=0
    for target,cp,qp in paths():
        ch,qh=header(cp),header(qp);common=set(ch)&set(qh);needed=[f for f in FIELDS if f in common]
        c=pd.read_csv(cp,float_precision='round_trip',usecols=needed,dtype={'prefecture':str});q=pd.read_csv(qp,float_precision='round_trip',usecols=needed,dtype={'prefecture':str})
        assert len(c)==1049 and len(q)==326 and not c.isna().any().any() and not q.isna().any().any()
        numeric=[f for f in needed if f!='prefecture']
        assert np.isfinite(c[numeric].to_numpy(float)).all() and np.isfinite(q[numeric].to_numpy(float)).all()
        eligible=[(name,cols) for name,cols in SETS if set(cols)<=common]
        cc,qc=csv_counters(cp,eligible),csv_counters(qp,eligible)
        for name,cols in SETS:
            missing_C=[f for f in cols if f not in ch];missing_Q=[f for f in cols if f not in qh];available=not missing_C and not missing_Q
            rec=dict(target=target,attribute_set=name,attribute_count=len(cols),C_rows=len(c),Q_rows=len(q),available=available,
                missing_C_fields='+'.join(missing_C),missing_Q_fields='+'.join(missing_Q),necessary_lower_bound=PRIVATE_N,
                n_C_in_Qsupport=None,n_Q_in_Csupport=None,shared_distinct_tuples=None,C_distinct_tuples=None,Q_distinct_tuples=None,
                weak_H_necessary_condition_failed=None,strong_H_forward_necessary_condition_failed=None,strong_H_either_necessary_condition_failed=None,
                passing_does_not_prove_H=True,algorithm_or_identity_established=False,attack_suitability_or_accuracy_established=False,method_selected=False)
            if available:
                fwd,rev,shared,cn,qn=directions(c,q,cols);shared_keys=set(cc[name])&set(qc[name])
                assert fwd==sum(cc[name][key] for key in shared_keys) and rev==sum(qc[name][key] for key in shared_keys)
                assert shared==len(shared_keys) and cn==len(cc[name]) and qn==len(qc[name])
                if name=='prefecture':assert fwd==old[target]
                rec.update(n_C_in_Qsupport=fwd,n_Q_in_Csupport=rev,shared_distinct_tuples=shared,C_distinct_tuples=cn,Q_distinct_tuples=qn,
                    weak_H_necessary_condition_failed=rev<PRIVATE_N,strong_H_forward_necessary_condition_failed=fwd<PRIVATE_N,
                    strong_H_either_necessary_condition_failed=fwd<PRIVATE_N or rev<PRIVATE_N)
                available_cases+=1
            else:unavailable_cases+=1
            records.append(rec)
    assert len(records)==264 and available_cases==220 and unavailable_cases==44
    table=out/'本戦22の固定12属性集合と双方向支持必要条件.csv';pd.DataFrame(records).to_csv(table,index=False,float_format='%.17g')
    write_json(out/'必要条件の仮説と証明範囲.json',proof(county))
    for p,h in plan['public_input_sha256'].items():assert sha(p)==h
    assert sha(COUNTY)==plan['county_condition_sha256'] and sha(__file__)==plan['code_sha256']
    record=dict(completed_jst=stamp(),code_sha256=sha(__file__),plan_sha256=sha(out/'事前固定条件.json'),
        public_input_sha256=plan['public_input_sha256'],county_condition_sha256=sha(COUNTY),output_sha256={p.name:sha(p) for p in out.iterdir()},
        targets=22,fixed_attribute_sets=12,available_attribute_sets=10,available_cases=available_cases,unavailable_cases=unavailable_cases,
        Counter_MultiIndex_max_count_error=0,independent_scalar_CSV_Counter_max_count_error=0,old_county_forward_max_count_error=0,
        common_fields_by_schema=['age','sex','prefecture','SBP','TG','HDL','ALT'],Q_missing_fixed_clinical_fields=['BMI','FPG'],
        private_target_id_or_membership_labels_used=False,B_used=False,attack_scores_used=False,method_selection=False,
        individual_rows_or_IDs_exported=False,NPZ_created=False,final_prose_created=False,external_action=False,
        python=sys.executable,python_version=platform.python_version())
    write_json(out/'実行記録.json',record);print(json.dumps({k:v for k,v in record.items() if k not in ('public_input_sha256','output_sha256')},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-plan',action='store_true');parser.add_argument('--output-dir',type=Path,default=HERE/'公開QとCの双方向支持必要条件_カテゴリ文字列対応結果');args=parser.parse_args()
    assert sys.dont_write_bytecode and Path(sys.executable).absolute()==ROOT/'.venv/bin/python'
    out=args.output_dir
    if args.write_plan:
        assert not out.exists();source=paths();schemas={str(p.resolve()):header(p) for _,c,q in source for p in (c,q)};qa=numerical_qa()
        assert all(set(FIELDS)-set(v)==({'BMI','FPG'} if 'aia_challenge_' in p else set()) for p,v in schemas.items())
        out.mkdir();plan=dict(created_jst=stamp(),code_sha256=sha(__file__),public_input_sha256={str(p.resolve()):sha(p) for _,c,q in source for p in (c,q)},
            county_condition_path=str(COUNTY),county_condition_sha256=sha(COUNTY),fixed_attribute_sets=[dict(name=n,columns=list(c)) for n,c in SETS],
            failed_attempt_sha256={str(p.resolve()):sha(p) for p in (FAILED,FAILED_OUT/'事前固定条件.json',FAILED_OUT/'停止記録.json')},
            public_schemas=schemas,missing_fixed_clinical_fields=['BMI','FPG'],missing_field_policy='Keep all 12 fixed sets for all 22; 44 missing cases count/fail flags blank.',
            exact_equality='typed observed numeric equality and exact prefecture string equality; pandas round_trip plus independent stdlib parser',
            computations='C occurrence count in Q unique support and Q occurrence count in C unique support; duplicates retained only in occurrence counts.',
            fixed_lower_bound=PRIVATE_N,numerical_qa=qa,source_rows='C1049 and Q326 from public inputs only',no_GT_B_score_guard=True,
            metadata_scope='necessary-condition failures only, no hypothesis confirmation, processing identity, attack suitability or method selection',
            membership_labels_used=False,B_used=False,attack_scores_used=False,individual_rows_or_IDs_exported=False,method_selection=False,final_prose_created=False,external_action=False)
        write_json(out/'事前固定条件.json',plan);print(json.dumps(dict(stage='plan_fixed',code_sha256=plan['code_sha256'],public_sources=len(plan['public_input_sha256']),qa=qa),ensure_ascii=False));return
    assert out.is_dir() and {p.name for p in out.iterdir()}=={'事前固定条件.json'}
    plan=json.loads((out/'事前固定条件.json').read_text());run(out,plan)


if __name__=='__main__':main()
