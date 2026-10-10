"""Frozen preliminary C, main-rule computable terms and conditional rigorous bounds.

No missing B is estimated. All 21 available C files are evaluated; cohort 21 stays missing.
Published decimal scores are conservatively treated as +/- 1e-9 intervals.
"""
from pathlib import Path
import ast, hashlib, json, copy, argparse
import numpy as np
import pandas as pd
import yaml

SOURCE_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--c-dir', type=Path, required=True, help='Directory containing the original preliminary C_*.csv files')
parser.add_argument('--out-dir', type=Path, required=True, help='New output directory')
args = parser.parse_args()
HERE = args.out_dir
HERE.mkdir(parents=True, exist_ok=True)
SRC = SOURCE_DIR
ROOT = args.c_dir
EPS = 1e-9
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: pd.read_csv(p, float_precision='round_trip')

def nodoc(n):
    n = copy.deepcopy(n)
    for x in ast.walk(n):
        if hasattr(x, 'body') and isinstance(x.body, list) and x.body and isinstance(x.body[0], ast.Expr) and isinstance(x.body[0].value, ast.Constant) and isinstance(x.body[0].value.value, str):
            x.body = x.body[1:]
    return ast.dump(n, include_attributes=False)

def run():
    trees = {phase: ast.parse((SRC/f'sources/{phase}_utility.py').read_text()) for phase in ['prelim','main']}
    funcs = {phase: {x.name:x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef))} for phase,tree in trees.items()}
    audits = {n: nodoc(x)==nodoc(funcs['main'][n]) for n,x in funcs['prelim'].items() if n in funcs['main']}
    (HERE/'executable_ast_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    for n in ['_prepare','_tvd','_propensity_scores','_pmse','_score_u_spec','_score_u_time','_score_u_rare','_no_fab','_aggregate']:
        assert audits[n], n
    cfg = yaml.safe_load((SRC/'sources/main-process-20260912_scoring_config.yaml').read_text())
    # Execute this exact official pure C-only function, without its unrelated heavy imports.
    ns = {'np':np,'pd':pd}
    exec(compile(ast.Module(body=[funcs['main']['_no_dup']], type_ignores=[]),str(SRC/'sources/main_utility.py'),'exec'),ns)
    official = read(SRC/'prelim_official_utility_all22.csv')
    manifest = json.loads((SOURCE_DIR/'C_input_manifest.json').read_text())
    mf = {f['filename']: f for f in manifest['files']}
    rows = []
    for old in official.to_dict('records'):
        k = int(old['team'])
        row = {'team':k,'team_name':old['team_name']}
        for col in ['U','U_gen','U_spec','U_rare','U_valid']:
            row['prelim_'+col] = old[col]
        p = ROOT/f'C_{k}.csv'
        if not p.exists():
            row['status']='missing_C_no_recomputation'
            rows.append(row)
            continue
        c = read(p)
        src = mf[p.name]
        assert sha(p) == src['sha256']
        assert c.shape==(1049,13) and not c.isna().any().any()
        no_dup, dup_detail = ns['_no_dup'](c,cfg)
        admin = (c.onset==0)&(c.death==0)
        hz = np.isclose(c.time.to_numpy(float),cfg['onset']['horizon_years'],atol=1e-2)
        cens_old = float(hz[admin].mean()) if admin.any() else 1.0
        cens_row = float(hz[admin].mean()) if admin.any() else 0.0
        p_c = float(hz.mean())
        # p_horizon(B) belongs to [0,1]; rate-match lies in [min(p_c,1-p_c),1].
        cens_lo = min(cens_row,min(p_c,1-p_c))
        cens_hi = cens_row
        # Invert the verified unchanged mean-2 definition. Interval includes ledger rounding.
        fab = 2*old['U_valid']-cens_old
        assert -2*EPS<=fab<=1+2*EPS, (k,fab)
        fab_lo = max(0.,fab-2*EPS)
        fab_hi = min(1.,fab+2*EPS)
        v_lo = (cens_lo+fab_lo+no_dup+0.)/4
        v_hi = (cens_hi+fab_hi+no_dup+1.)/4
        # Main Ugen is the same prelim min-3 plus corr and tail. Uspec/Urare unchanged.
        u_hi = min(old['U_gen']+EPS,old['U_spec']+EPS,old['U_rare']+EPS,v_hi,1.)
        row.update(status='bounded_missing_B_and_reference',C_sha256=sha(p),C_source_url=src['source_url'],n_rows=len(c),
            n_unique=len(c.drop_duplicates()),n_duplicate_extra=dup_detail['n_dup_rows'],dup_rate=dup_detail['dup_rate'],
            main_no_dup_exact=no_dup,n_admin_censored=int(admin.sum()),prelim_cens_exact=cens_old,
            main_cens_rowwise_exact=cens_row,C_horizon_rate=p_c,main_cens_lower=cens_lo,main_cens_upper=cens_hi,
            no_fab_from_ledger=fab,no_fab_lower=fab_lo,no_fab_upper=fab_hi,
            main_novel_lower=0.,main_novel_upper=1.,main_U_valid_lower=v_lo,main_U_valid_upper=v_hi,
            main_U_gen_lower=0.,main_U_gen_upper=min(1.,old['U_gen']+EPS),
            main_U_spec_lower=max(0.,old['U_spec']-EPS),main_U_spec_upper=min(1.,old['U_spec']+EPS),
            main_U_rare_lower=max(0.,old['U_rare']-EPS),main_U_rare_upper=min(1.,old['U_rare']+EPS),
            main_U_lower=0.,main_U_upper=u_hi,delta_U_lower=-(old['U']+EPS),delta_U_upper=u_hi-max(0.,old['U']-EPS),
            main_U_valid_C_only_upper=(cens_row+no_dup+2.)/4,
            main_U_C_only_upper=(cens_row+no_dup+2.)/4,
            C_only_guaranteed_U_drop_at_least=max(0.,old['U']-EPS-(cens_row+no_dup+2.)/4),
            guaranteed_U_drop_at_least=max(0.,old['U']-EPS-u_hi),ledger_tolerance=EPS,
            conditional_on='same preliminary B, original reference, A_bg; official ledger matched same C and retained formulas; ignore finite-precision parser effects on retained facets')
        rows.append(row)
    df=pd.DataFrame(rows)
    df.to_csv(HERE/'all22_utility_bounds.csv',index=False,float_format='%.17g')
    df[df.status!='missing_C_no_recomputation'].to_csv(HERE/'all21_utility_bounds.csv',index=False,float_format='%.17g')
    cols=['team','team_name','prelim_U','main_U_lower','main_U_upper','guaranteed_U_drop_at_least','n_duplicate_extra','main_no_dup_exact','n_admin_censored','main_cens_rowwise_exact','main_U_valid_upper','status']
    df[cols].to_csv(HERE/'utility_comparison_compact.csv',index=False,float_format='%.10g')
    assumptions={'counterfactual':'Keep each original preliminary C, B, utility reference, preliminary A_bg fixed; replace scoring code and config with main-process-20260912.',
       'not_actual_main_leaderboard':True,'not_new_main_cohorts':True,'all21_exact_main_U_available':False,
       'missing_B_and_reference_teams':[int(t) for t in df.loc[(df.team!=5)&(df.team!=21),'team']],
       'missing_C_team':21,'ledger_tolerance':EPS,'bounds_do_not_estimate_unknown_B':True,
       'no_fab_identity':'2 * preliminary U_valid - preliminary C-only censoring score',
       'main_Uvalid_bounds':'(bounds of new censor + no_fab + exact no_dup + novel in [0,1])/4',
       'main_U_upper':'min(preliminary U_gen, preliminary U_spec, preliminary U_rare, new U_valid upper); each ledger value enlarged by tolerance',
       'main_U_lower':'0, because uncomputed added correlation/tail constraints can in principle bind at zero',
       'main_U_C_only_upper':'(main censor-rowwise + exact no_dup + 2)/4; independent of any transferred ledger facet, B, and original utility reference',
       'parser':'main official read_scoring_csv uses pandas float_precision=round_trip; preliminary adapter used default reader',
       'runtime':'Exact C-only official _no_dup function executed by AST; no substitute scoring formulas for this term',
       'ledger_source_sha256':sha(SRC/'prelim_official_utility_all22.csv'),
       'code_sha256':{phase:sha(SRC/f'sources/{phase}_utility.py') for phase in ['prelim','main']}}
    (HERE/'bounds_provenance.json').write_text(json.dumps(assumptions,ensure_ascii=False,indent=2)+'\n')
    print(df[cols].to_string(index=False))

if __name__=='__main__':run()
