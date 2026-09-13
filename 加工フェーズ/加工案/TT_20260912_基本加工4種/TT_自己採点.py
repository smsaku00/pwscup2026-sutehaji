#!/usr/bin/env python3
"""生成と完全に分離した公式キットの自己採点用アダプター。v1.0 / 2026-09-12.
--kit: main-process-20260912 kit root containing starter/src
--dist: team distribution with B_self, utility_ref, schema, A_bg
--config: supplied kit participant_data/scoring_config.yaml
--c: generated CSV(s), --out: report JSON (outside distribution directory)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['kit','dist','config','out']:
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--c',required=True,nargs='+',type=Path)
    a=p.parse_args()
    if a.out.resolve().is_relative_to(a.dist.resolve()) or a.out.resolve() in [x.resolve() for x in a.c]:
        p.error('配布物フォルダ・入力CSVを出力先に指定できません')
    src=a.kit/'starter/src'
    if not src.is_dir():
        p.error('--kitにstarter/srcがありません')
    sys.path.insert(0,str(src.resolve()))
    from pwscup2026.kit.selfscore import load_local_reference,self_score
    from pwscup2026.common.submission_schema import read_scoring_csv,validate_c_submission
    from pwscup2026.scoring.result import json_safe
    results=[]
    with tempfile.TemporaryDirectory(prefix='TT-selfscore-') as td:
        staged=Path(td)
        for name in ['B_self.csv','utility_ref.json','schema.json','A_bg.csv']:
            shutil.copyfile(a.dist/name,staged/name)
        shutil.copyfile(a.config,staged/'scoring_config.yaml')
        b,ref,cfg,info=load_local_reference(staged)
        for path in a.c:
            c=read_scoring_csv(path)
            v=validate_c_submission(c,cfg,n_expected=len(b))
            if not v.ok:
                p.error('; '.join(v.errors))
            start=time.perf_counter()
            score=self_score(c,b,ref,cfg)
            score.update({'csv':path.name,'seconds':time.perf_counter()-start,'evaluation':'本戦公式キットによるローカル自己採点（CodaBench公式記録とは別）','AIA':None,'Anon':None,'protection':None})
            results.append(json_safe(score))
    a.out.write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(a.out)

if __name__=='__main__':
    main()
