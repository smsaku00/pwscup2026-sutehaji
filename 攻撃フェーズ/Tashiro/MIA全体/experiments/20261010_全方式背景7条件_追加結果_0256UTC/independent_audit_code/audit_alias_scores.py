"""Numerical regression of separately source-proven aliases, not discovery by equality."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;T=Path('攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件')
aliases=pd.read_csv(R/'inventory/cross_panel_semantic_aliases.csv')
BG=['starter','prelim','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']
results=[]
for bg in BG:
    cache={}
    def get(cid):
        if cid in cache:return cache[cid]
        panel,phase,protocol,family,inference,method=cid.split('|')
        base=Path('/workspace/shared/mia_all_bg_'+panel)/T/panel
        if panel=='available':
            out=base/f'results_{phase}{"_primary" if protocol=="primary" else ""}'
            teams=json.loads((out/'fixed_protocol.json').read_text())['inference_targets'][inference]
            with np.load(out/f'predictions_{bg}_{inference}.npz') as z: matrix=z[method]
        else:
            records=json.loads((base/phase/bg/'prediction_hashes.json').read_text())
            indexes=[i for i,r in enumerate(records) if (r['family'],r['inference'],r['method'])==(family,inference,method)]
            assert len(indexes)==1,(cid,indexes)
            i=indexes[0];record=records[i];teams=record['teams']
            path=Path('/workspace/shared/mia_'+panel+'_cache')/f'{phase}_{bg}.npz'
            with np.load(path) as z:matrix=z[str(i) if panel=='joint' else record['key']]
        order=np.argsort(teams);cache[cid]=(sorted(teams),matrix[:,order]);return cache[cid]
    for x in aliases.to_dict('records'):
        ta,a=get(x['canonical_config_id']);tb,b=get(x['original_config_id'])
        assert ta==tb
        error=float(np.max(abs(a-b)))
        assert error<1e-12,(bg,x,error)
        results.append({'background':bg,**x,'max_abs_score_difference':error,'bitwise_equal':bool(np.array_equal(a,b))})
pd.DataFrame(results).to_csv(R/'inventory/cross_panel_alias_score_regression.csv',index=False)
summary={'status':'passed','source_proven_aliases':len(aliases),'backgrounds':7,'comparisons':len(results),'max_abs_score_difference':max(x['max_abs_score_difference'] for x in results),'all_bitwise_equal':all(x['bitwise_equal'] for x in results),'note':'Equality only validates pre-existing source proofs. No alias was inferred from observed scores.'}
(R/'inventory/cross_panel_alias_score_regression.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
