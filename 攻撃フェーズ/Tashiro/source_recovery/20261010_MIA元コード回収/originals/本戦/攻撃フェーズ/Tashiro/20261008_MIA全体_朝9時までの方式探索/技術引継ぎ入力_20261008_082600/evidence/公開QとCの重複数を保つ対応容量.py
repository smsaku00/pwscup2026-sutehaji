"""同じ公開Q/C固定12集合について重複数を保つ対応容量とprivate163不足下限を診断。

M=sum_tuple min(countQ,countC)。個人tuple/ID/scoreは保存しない。
旧成功/停止版を保持し、必要条件の通過を実private対応の証明にはしない。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'公開QとCの双方向支持必要条件_カテゴリ文字列対応.py'
spec=importlib.util.spec_from_file_location('fixed_public_support_capacity',SOURCE)
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
PREVIOUS=HERE/'公開QとCの双方向支持必要条件_カテゴリ文字列対応結果'
PREVIOUS_TABLE=PREVIOUS/'本戦22の固定12属性集合と双方向支持必要条件.csv'


def frozen_hashes():
    files=[SOURCE,HERE/'公開QとCの双方向支持必要条件.py',
        *(p for p in PREVIOUS.iterdir() if p.is_file()),
        *(p for p in (HERE/'公開QとCの双方向支持必要条件結果').iterdir() if p.is_file())]
    return {str(p.resolve()):old.sha(p) for p in sorted(files)}


def capacity(q,c):return sum(min(n,c.get(t,0)) for t,n in q.items())


def bipartite_maximum(q,c):
    # Independent exhaustive matching of labelled occurrences, not a Counter formula.
    @lru_cache(None)
    def best(i,used):
        if i==len(q):return 0
        value=best(i+1,used)
        for j,label in enumerate(c):
            if not used&(1<<j) and label==q[i]:value=max(value,1+best(i+1,used|(1<<j)))
        return value
    return best(0,0)


def numerical_qa():
    sequences=[tuple(x) for n in range(4) for x in itertools.product(('a','b'),repeat=n)]
    cases=0
    for q in sequences:
        for c in sequences:
            assert capacity(Counter(q),Counter(c))==bipartite_maximum(q,c);cases+=1
    q=('a',)*10+('b',);c=('a',)+('b',)*10
    qc,cc=Counter(q),Counter(c);m=capacity(qc,cc)
    assert m==bipartite_maximum(q,c)==2
    assert sum(n for t,n in qc.items() if t in cc)==sum(n for t,n in cc.items() if t in qc)==11
    assert max(0,5-11)==0 and max(0,5-m)==3
    return dict(exhaustive_binary_sequence_matching_cases=cases,including_empty_and_duplicate=True,
        cross_duplicate_example=dict(Q_occurrences=11,C_occurrences=11,both_directional_support_counts=11,capacity=2,
            illustrative_private_count=5,support_deficit_lower_bound=0,distinct_deficit_lower_bound=3),
        Counter_enumerated_bipartite_maximum_error=0)


def proof(previous):
    return dict(kind='fixed_public_multiset_capacity_necessary_conditions_not_final_report',
        official_definition=previous['official_definition'],official_definition_source_sha256=previous['official_definition_source_sha256'],
        private_target_occurrences=163,definition='M=sum over observed tuples min(countQ(tuple),countC(tuple))',
        maximum_matching_proof='Equal-tuple edges form disjoint complete bipartite blocks. Each block admits min(countQ,countC) distinct occurrence matches; summing yields the maximum Q-to-C matching capacity.',
        strong_H=previous['strong_H'],strong_necessary_condition='M >= 163',
        strong_proof='Any distinct tuple-preserving matching of all private163 occurrences is a matching of a subset of all Q occurrences. Its size cannot exceed the all-Q maximum capacity M.',
        weak_unsupported_private_lower_bound='max(0,163-n_Q_in_Csupport)',
        weak_lower_bound_proof='At most n_Q_in_Csupport of all Q occurrences can be supported in C; therefore at most that many private occurrences can be supported.',
        distinct_unmatched_private_lower_bound='max(0,163-M)',
        distinct_lower_bound_proof='At most M Q occurrences can be matched to distinct C occurrences with the same tuple. Any private subset has matching size at most M.',
        bounds_do_not_identify_private_occurrences=True,capacity_pass_does_not_prove_actual_private_matching_or_H=True,
        two_support_bounds_weaker_than_capacity='M <= min(n_C_in_Qsupport,n_Q_in_Csupport); both support bounds can pass while capacity fails.',
        example_interpretation='A reverse support count of 99 implies at least64 of the private163 observed tuples are absent from C support; no individual is identified.',
        univariate_column_permutation=previous['univariate_column_permutation'],joint_column_permutation=previous['joint_column_permutation'],
        equality=previous['equality'],missing_fields='BMI/FPG are absent from Q: same44 cases remain unavailable with all new count/bound/fail fields blank.',
        resampling_transformation_correspondence_loss_not_distinguished=True,patient_identity_or_processing_algorithm_not_established=True,
        attack_suitability_or_accuracy_not_established=True,private_ID_or_membership_labels_used=False,B_used=False,attack_scores_used=False,
        method_selection=False,individual_tuples_or_IDs_exported=False,final_prose_created=False,external_action=False)


def run(out,plan):
    assert old.sha(__file__)==plan['code_sha256'] and frozen_hashes()==plan['frozen_previous_sha256']
    with PREVIOUS_TABLE.open(newline='') as f:previous=list(csv.DictReader(f))
    indexed={(int(r['target']),r['attribute_set']):r for r in previous};assert len(indexed)==264
    records=[];cases=0;unavailable=0
    for target,cp,qp in old.paths():
        common=set(old.header(cp))&set(old.header(qp));eligible=[(name,cols) for name,cols in old.SETS if set(cols)<=common]
        cc,qc=old.csv_counters(cp,eligible),old.csv_counters(qp,eligible)
        for name,cols in old.SETS:
            rec=dict(indexed[target,name]);rec.update(maximum_distinct_Q_C_tuple_capacity=None,
                strong_capacity_necessary_condition_failed=None,private_unsupported_tuple_occurrences_lower_bound=None,
                private_unmatched_distinct_occurrences_lower_bound=None,capacity_pass_does_not_prove_private_matching=True)
            if rec['available']=='True':
                shared=set(qc[name])&set(cc[name]);forward=sum(cc[name][t] for t in shared);reverse=sum(qc[name][t] for t in shared)
                assert forward==int(rec['n_C_in_Qsupport']) and reverse==int(rec['n_Q_in_Csupport'])
                assert len(shared)==int(rec['shared_distinct_tuples']) and len(cc[name])==int(rec['C_distinct_tuples']) and len(qc[name])==int(rec['Q_distinct_tuples'])
                m=capacity(qc[name],cc[name]);weak=max(0,163-reverse);strong=max(0,163-m)
                assert 0<=m<=min(forward,reverse) and 0<=weak<=strong<=163
                rec.update(maximum_distinct_Q_C_tuple_capacity=m,strong_capacity_necessary_condition_failed=m<163,
                    private_unsupported_tuple_occurrences_lower_bound=weak,private_unmatched_distinct_occurrences_lower_bound=strong)
                cases+=1
            else:
                assert name in ('BMI','FPG');unavailable+=1
            records.append(rec)
    assert len(records)==264 and cases==220 and unavailable==44
    pd.DataFrame(records).to_csv(out/'本戦22の固定12集合の最大対応容量とprivate不足下限.csv',index=False,float_format='%.17g')
    previous_proof=json.loads((PREVIOUS/'必要条件の仮説と証明範囲.json').read_text())
    old.write_json(out/'対応容量と不足下限の証明範囲.json',proof(previous_proof))
    assert frozen_hashes()==plan['frozen_previous_sha256'] and old.sha(__file__)==plan['code_sha256']
    for p,h in plan['public_input_sha256'].items():assert old.sha(p)==h
    record=dict(completed_jst=old.stamp(),code_sha256=old.sha(__file__),support_code_sha256={str(SOURCE):old.sha(SOURCE)},
        plan_sha256=old.sha(out/'事前固定条件.json'),public_input_sha256=plan['public_input_sha256'],frozen_previous_sha256=plan['frozen_previous_sha256'],
        output_sha256={p.name:old.sha(p) for p in out.iterdir()},rows=264,available_cases=220,unavailable_cases=44,
        previous_direction_and_distinct_count_max_error=0,previous_fields_preserved=True,Counter_capacity_formula_fixed=True,
        private_ID_or_membership_labels_used=False,B_used=False,attack_scores_used=False,method_selection=False,
        individual_tuples_or_IDs_exported=False,NPZ_created=False,final_prose_created=False,external_action=False,python=sys.executable)
    old.write_json(out/'実行記録.json',record);print(json.dumps({k:v for k,v in record.items() if not k.endswith('sha256')},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-plan',action='store_true');parser.add_argument('--output-dir',type=Path,default=HERE/'公開QとCの重複数を保つ対応容量結果');args=parser.parse_args()
    assert sys.dont_write_bytecode and Path(sys.executable).absolute()==old.ROOT/'.venv/bin/python'
    out=args.output_dir
    if args.write_plan:
        assert not out.exists();qa=numerical_qa();source=old.paths();out.mkdir()
        plan=dict(created_jst=old.stamp(),code_sha256=old.sha(__file__),support_code_sha256={str(SOURCE):old.sha(SOURCE)},
            public_input_sha256={str(p.resolve()):old.sha(p) for _,c,q in source for p in (c,q)},frozen_previous_sha256=frozen_hashes(),
            fixed_attribute_sets=[dict(name=n,columns=list(c)) for n,c in old.SETS],fixed_private_count=163,
            capacity='sum_tuple min(countQ,countC)',weak_deficit='max(0,163-n_Q_in_Csupport)',distinct_deficit='max(0,163-M)',
            same_sources_and_missing_policy_as_previous=True,rows=264,available_cases=220,unavailable_cases=44,
            exact_typed_equality='prefecture exact string, numeric columns round_trip numeric equality; no tolerance',
            numerical_qa=qa,previous264_available_and_direction_reproduction_barrier=True,passing_not_proof_of_actual_private_matching_or_H=True,
            private_ID_or_membership_labels_used=False,B_used=False,attack_scores_used=False,method_selection=False,
            individual_tuples_or_IDs_exported=False,NPZ_created=False,final_prose_created=False,external_action=False)
        old.write_json(out/'事前固定条件.json',plan);print(json.dumps(dict(stage='plan_fixed',code_sha256=plan['code_sha256'],qa=qa),ensure_ascii=False));return
    assert out.is_dir() and {p.name for p in out.iterdir()}=={'事前固定条件.json'}
    run(out,json.loads((out/'事前固定条件.json').read_text()))


if __name__=='__main__':main()
