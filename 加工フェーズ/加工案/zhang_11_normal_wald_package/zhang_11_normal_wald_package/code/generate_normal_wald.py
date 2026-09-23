"""Zhang 11: Zhang04 with Wald-calibrated outcomes in the normal stratum."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from generate import generate as wald_generate
from generate_three import generate as original_generate
from shared_methods import public_outliers


def generate(b, a, schema):
    original, _ = original_generate(b, a, schema, 'zhang_04_copula_bgtime')
    calibrated, _ = wald_generate(b, a, schema)
    same = original[['age','BMI','SBP','TG','HDL','ALT','FPG']].equals(
        calibrated[['age','BMI','SBP','TG','HDL','ALT','FPG']])
    if not same:
        raise ValueError('Base and Wald covariates do not match row by row')
    rare = ~public_outliers(original, schema)
    result = original.copy()
    result.loc[rare, ['time','onset','death']] = calibrated.loc[
        rare, ['time','onset','death']].to_numpy()
    return result, {'method':'zhang_11_normal_wald', 'normal_outcome_rows':int(rare.sum()),
                    'normal_outcome_rows_changed':int((original.loc[rare,'time'] !=
                                                     calibrated.loc[rare,'time']).sum())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['b','a-bg','schema','out']:
        p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args()
    b=pd.read_csv(a.b,float_precision='round_trip')
    bg=pd.read_csv(a.a_bg,float_precision='round_trip')
    schema=json.loads(a.schema.read_text(encoding='utf-8'))
    c, meta=generate(b,bg,schema)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    data=c.to_csv(index=False,float_format='%.17g',lineterminator='\n').encode()
    a.out.write_bytes(data)
    meta['C_sha256']=hashlib.sha256(data).hexdigest()
    print(json.dumps(meta,ensure_ascii=False))


if __name__=='__main__':main()
