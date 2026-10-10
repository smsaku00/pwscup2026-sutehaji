"""Merge the exact local self5 evaluation into the all-team bounds table."""
from pathlib import Path
import pandas as pd
import json
H=Path(__file__).resolve().parent
df=pd.read_csv(H/'all22_utility_bounds.csv',float_precision='round_trip')
p=json.loads((H/'self5_prelim_breakdown.json').read_text())
m=json.loads((H/'self5_main_breakdown.json').read_text())
i=df.index[df.team==5][0]
for metric in ['U','U_gen','U_spec','U_rare','U_valid']:
    v=m['score'] if metric=='U' else m['facets'][metric]
    for suffix in ['point','lower','upper']:df.loc[i,f'main_{metric}_{suffix}']=v
df.loc[i,'delta_U_lower']=df.loc[i,'delta_U_upper']=m['score']-p['score']
df.loc[i,'guaranteed_U_drop_at_least']=p['score']-m['score']
df.loc[i,'status']='local_recomputed_official_code_and_locked_dependencies'
for term,key in [('main_cens','censoring_realism_rate'),('main_novel','novel'),('no_fab','no_fab')]:
    for suffix in ['lower','upper']:df.loc[i,f'{term}_{suffix}']=m['detail']['U_valid'][key]
df.to_csv(H/'all22_utility_final.csv',index=False,float_format='%.17g')
df[df.team!=21].to_csv(H/'all21_utility_final.csv',index=False,float_format='%.17g')
print('Merged self5; 20 other available teams remain bounded, cohort21 remains missing.')
