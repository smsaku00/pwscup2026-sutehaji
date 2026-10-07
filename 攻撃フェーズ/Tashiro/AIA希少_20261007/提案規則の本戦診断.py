"""固定規則の本戦での変更件数だけを確認する。提出回答は保存しない。

推定希少上位52件のうち、C距離<全他Cの最近傍距離の中央値ならNNのtime、
そうでなければ11年。それ以外の行と、公式結果の強いチーム9は現回答を保持。
正解は未知なので得点は計算しない。自チームBも読み込まない。
"""
from __future__ import annotations
import 分析 as core
import argparse, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import 希少候補の学習 as raremod


def run(root):
    pre,bg=core.datasets(root,'prelim')
    model=raremod.make_model().fit(np.vstack([raremod.features(v['q']) for v in pre]),np.concatenate([v['truth'].is_rare.to_numpy(bool) for v in pre]))
    cases,mbg=core.datasets(root,'main');current=core.read(root/'本戦/攻撃フェーズ/Tashiro/20261006_第2回提出データ統合/入力/AIA/F_aia.csv');rows=[]
    for i,v in enumerate(cases):
        if v['team']==28:continue
        q=v['q'];base=current[current.target_k.eq(v['team'])].set_index('challenge_row_id').loc[q.challenge_row_id,'time_hat'].to_numpy()
        prob=model.predict_proba(raremod.features(q))[:,1];selected=raremod.top(prob,52)
        foreign=[];times=None;selfdist=None
        for j,w in enumerate(cases):
            d=core.cost(q,w['c'],mbg,5,'sex')
            if i==j:
                nn=d.argmin(1);selfdist=d[np.arange(326),nn];times=w['c'].time.to_numpy()[nn]
            else:foreign.append(d.min(1))
        reference=np.median(np.column_stack(foreign),axis=1);pass_=selfdist<=reference
        pred=np.where(selected,np.where(pass_,times,11.),base)
        if v['team']==9:pred=base.copy()
        changed=np.abs(pred-base)>1e-10
        assert not changed[~selected].any()
        if v['team']==9:assert not changed.any()
        rows.append(dict(team=v['team'],estimated_rare_candidates=52,changed=int(changed.sum()),
                         changed_to_11=int((changed&(pred==11)).sum()),changed_to_time=int((changed&(pred!=11)).sum()),
                         candidate_reference_test_pass=int(pass_[selected].sum()),outside_candidates_changed=0,
                         target9_preserved=v['team']==9,main_rare_truth_known=False,main_score_computed=False))
    out=core.HERE/'集計/提案規則_本戦の変更件数.csv';pd.DataFrame(rows).to_csv(out,index=False)
    core.record('提案規則の本戦診断',dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        total_changed=sum(v['changed'] for v in rows),target9_changed=0,outside_candidates_changed=0,main_truth_used=False,
        proposed_rule_frozen_top=52,reference_threshold=0,submission_csv_saved=False))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=core.ROOT);run(p.parse_args().root.resolve())
