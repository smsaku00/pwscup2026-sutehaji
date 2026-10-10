"""保存物を変更せず入力・出力ハッシュ、全セル、公式MIA形式を再確認。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from 分析 import HERE,ROOT,OUT,validator


def main():
    for name,root in [('入力SHA256.json',ROOT),('出力SHA256.json',OUT)]:
        hashes=json.loads((OUT/name).read_text())
        for path,expected in hashes.items():
            actual=hashlib.sha256((root/path).read_bytes()).hexdigest()
            assert actual==expected, path
        print(f'{name}: {len(hashes)}件一致')
    config=json.loads((OUT/'計算条件.json').read_text())
    f=pd.read_csv(OUT/'F_mia.csv',float_precision='round_trip')
    raw=pd.read_csv(OUT/'本戦_生スコア.csv',float_precision='round_trip')
    private=pd.read_csv(OUT/'本戦_非会員163人.csv')
    rare_rows=pd.read_csv(OUT/'本戦_希少24人の順位.csv')
    rare_ids=set(rare_rows.record_id)
    expected_teams=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,25,26,27]
    assert f.shape==(1049,23)
    assert f.columns.tolist()==['record_id',*map(str,expected_teams)]
    assert f.record_id.tolist()==list(range(1049))
    assert len(private)==163 and private.is_rare.sum()==13 and len(rare_ids)==24
    pm=f.record_id.isin(private.record_id).to_numpy()
    rm=f.record_id.isin(rare_ids).to_numpy()
    assert (f.iloc[pm,1:].to_numpy()==0).all()
    assert int((f.iloc[:,1:]==0).sum().sum())==163*22
    for team in expected_teams:
        rows=raw[raw.team==team].set_index('record_id').loc[f.record_id]
        # pandas順位で提出全セルを独立再構成する。
        expected=np.zeros(len(f))
        n=~pm&~rm;r=~pm&rm
        expected[n]=.5+.5*(.1+.8*rows.loc[n,'mixed7'].rank(method='min').to_numpy()/n.sum())
        expected[r]=.5*(.9-.8*(rows.loc[r,'mixed9'].rank(method='min',ascending=False).to_numpy()-1)/r.sum())
        np.testing.assert_array_equal(f[str(team)].to_numpy(),expected)
        assert f.loc[n,str(team)].min()>f.loc[r,str(team)].max()
    ref={'mia_columns':{'target_columns':[*map(str,expected_teams),'28']},'team_id':28,'attacker_record_ids':set(range(1049))}
    result=validator.validate_mia_submission(f,{},ref)
    assert result.ok,result.errors
    print('23,078セルの独立再構成一致、非会員3,586セルが0、自列28なし、公式MIA検査成功')
    print('本戦精度は未評価。AIA・token・ZIP・サーバ提出は今回の検証対象外。')


if __name__=='__main__':
    main()
