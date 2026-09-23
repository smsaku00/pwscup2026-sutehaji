"""Adapter for pwscup_reusable_eval; strips B_self's evaluation-only label."""
import argparse
import json
from pathlib import Path
import pandas as pd

from generate import generate as generate_11
from generate_normal_wald import generate as generate_13
from generate_three import generate as generate_original


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--b', required=True, type=Path)
    p.add_argument('--a-bg', required=True, type=Path)
    p.add_argument('--schema', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--method', choices=['z04', 'full_wald', 'z11'], required=True)
    p.add_argument('--wald-weight', type=float, default=.3)
    a = p.parse_args()
    b = pd.read_csv(a.b, float_precision='round_trip').drop(columns=['is_rare'], errors='ignore')
    bg = pd.read_csv(a.a_bg, float_precision='round_trip')
    schema = json.loads(a.schema.read_text(encoding='utf-8'))
    if a.method == 'full_wald':
        c, _ = generate_11(b, bg, schema, wald_weight=a.wald_weight)
    elif a.method == 'z11':
        c, _ = generate_13(b, bg, schema)
    else:
        c, _ = generate_original(b, bg, schema, 'zhang_04_copula_bgtime')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    c.to_csv(a.out, index=False, float_format='%.17g', lineterminator='\n')


if __name__ == '__main__':
    main()
