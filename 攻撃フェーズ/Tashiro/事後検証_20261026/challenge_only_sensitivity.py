#!/usr/bin/env python3
"""Cohort 28 challenge-only, clinically interpretable fixed rules.

Uses public aia_challenge_28.csv to assign a constant to feature-defined groups;
labels/times are used exclusively for retrospective evaluation. If the
challenge CSV is unavailable locally, previously extracted, fixed rule masks
are used to evaluate the exact 2026-10-26 version only. The masks are an
unaltered encoding of rule matches, not fit to labels.
"""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from posthoc_analysis import HERE,TRUTHS,aia_eval

K=28
ROOT=HERE.parent.parent.parent/'公式データ'/'PWSCup2026_final_attack' # this path is valid in the repository only
TARGET=ROOT/'aia_challenge_28.csv'
MASKS={
'age_ge60':'108cd1891202243a6e8801e84a80a424602080c08030120062f21871f5082a5c020660cab81a084022',
'age_ge65':'108cd0880202040a4e8001e80a80a424602080408010120022701070e5080a54020660c02810004022',
'age_lt50':'3710454884540c1014206110533585b9dda32193fc681e69d0cc6080226950144c1032140c5b127d4',
'sbp_ge130':'39c853ac22240224f608f9627a8806cde231c04818b93f31e4e4ab732828b908951b84beb18fdac204',
'sbp_ge140':'29c041a4002002008400e962788800852011c0081818063160c0a23328003908800800a4a00380c204',
'tg_ge150':'202328aa0142052008005042940816040084413b10221ca2ac24c2a6b8f6880b104828028500108808',
'tg_ge200':'2003202201400120080050428000100000844111002018a20020422218d4880b004000028400108800',
'hdl_lt40':'8400081c80000200000400000100000004000002000000120000000000000000001000800000080',
'hdl_lt50':'148f4468abcbf01826207e563d221fa29584c4b821a4816b8f79c2c27cdcd8ec13c059a66a402b98a9',
'alt_ge30':'c6100a6e80000c009800c010c4818049c04500908e819b2008caa0381bc9000200906828020d2848f',
'alt_ge40':'80000a28000004009800400080008040404500908c808a200042a02808c900000090480800010848e',
'sex_0':'2000000000080000028004080000001400210000008000000088000001000000000040000004',
'risk2of4':'208841a8014204202c004162588814042000c0081030162060e08233b800280800082082a80200c000',
}


def evaluate_rows():
    # In the checked-in repository use the original published challenge for rule assignment.
    if TARGET.exists():
        with TARGET.open(encoding='utf-8-sig',newline='') as f:obs=list(csv.DictReader(f))
        if len(obs)!=326: raise ValueError('Unexpected challenge size')
        def rules(r):
            age=int(r['age']);sbp=float(r['SBP']);tg=float(r['TG']);hdl=float(r['HDL']);alt=float(r['ALT']);sex=int(r['sex'])
            risk=sum([age>=60,sbp>=140,tg>=150,hdl<40])
            return {'age_ge60':age>=60,'age_ge65':age>=65,'age_lt50':age<50,
                    'sbp_ge130':sbp>=130,'sbp_ge140':sbp>=140,'tg_ge150':tg>=150,
                    'tg_ge200':tg>=200,'hdl_lt40':hdl<40,'hdl_lt50':hdl<50,
                    'alt_ge30':alt>=30,'alt_ge40':alt>=40,'sex_0':sex==0,'risk2of4':risk>=2}
        return {name:{int(r['challenge_row_id']) for r in obs if rules(r)[name]} for name in MASKS}
    return {name:{j for j in range(326) if (int(mask,16)>>j)&1} for name,mask in MASKS.items()}


def main():
    flagged=evaluate_rows()
    ids={int(r['challenge_row_id']) for r in TRUTHS[28]}
    result=[]
    for key,selection in flagged.items():
        assert selection<=ids
        for default in [9.5,10.]:
            for override in [0.,5.,8.]:
                predictions={i:(override if i in selection else default) for i in ids}
                z=aia_eval(K,predictions)
                member_selected=sum(r['label']=='member' and int(r['challenge_row_id']) in selection for r in TRUTHS[K])
                control_selected=len(selection)-member_selected
                result.append({'feature_rule':key,'n_selected':len(selection),
                               'member_selected':member_selected,'control_selected':control_selected,
                               'default_time':default,'selected_time':override,'aia_score':z['aia_floored'],
                               'aia_all_R':z['aia_all'],'aia_rare_R':z['aia_rare'],
                               'p_member':z['p_m'],'p_control':z['p_c']})
    result.sort(key=lambda r:-r['aia_score'])
    out=HERE/'self_target28_results'/'aia28_feature_only_rules.csv'
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=result[0]);w.writeheader();w.writerows(result)
    print('best rules')
    for r in result[:20]:print(r)
    print('default 9.5 & selected=0')
    for r in result:
        if r['default_time']==9.5 and r['selected_time']==0:
            print(r['feature_rule'],round(r['aia_score'],6),r['member_selected'],r['control_selected'])

if __name__=='__main__':main()