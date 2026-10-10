"""Run both official utility implementations on unchanged prelim team5 B/C/reference."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
SOURCE_DIR=Path(__file__).resolve().parent
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--b',type=Path,required=True,help='Original self-team B, including is_rare')
parser.add_argument('--c',type=Path,required=True,help='Original preliminary C_5.csv')
parser.add_argument('--background',type=Path,required=True,help='Original preliminary A_bg.csv')
parser.add_argument('--reference',type=Path,required=True,help='Original team5 utility_ref.json; intentionally not bundled')
parser.add_argument('--out-dir',type=Path,required=True)
parser.add_argument('--dependency-dir',type=Path,help='Optional directory installed from requirements-kit.txt')
args=parser.parse_args()
HERE=args.out_dir
HERE.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(SOURCE_DIR/'official_modules'))
if args.dependency_dir:sys.path.insert(0,str(args.dependency_dir))
import json,hashlib,time,platform,importlib.metadata,warnings
import numpy as np
import pandas as pd
import yaml
from prelim.scoring import utility as p
from main.scoring import utility as m
from prelim.scoring.result import ReportLevel as PLevel
from main.scoring.result import ReportLevel as MLevel
from main.common.submission_schema import read_scoring_csv

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(np.ndarray,list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.generic):return safe(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(n,d):(HERE/n).write_text(json.dumps(safe(d),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def run():
    t=time.time()
    bp=args.b
    cp=args.c
    ap=args.background
    # Honor each phase official parser: default for preliminary, round_trip for main.
    braw=pd.read_csv(bp);b=braw.drop(columns=['is_rare'])
    c=pd.read_csv(cp);c=c.sort_values(by=list(c.columns),kind='mergesort').reset_index(drop=True);c.insert(0,'record_id',range(len(c)))
    abg=read_scoring_csv(ap)
    bm=read_scoring_csv(bp).drop(columns=['is_rare'])
    cm=read_scoring_csv(cp);cm=cm.sort_values(by=list(cm.columns),kind='mergesort').reset_index(drop=True);cm.insert(0,'record_id',range(len(cm)))
    r=json.loads(args.reference.read_text())
    r['mu']=np.asarray(r['mu']);r['sigma']=np.asarray(r['sigma']);r['b_is_rare']=braw.is_rare.to_numpy(bool)
    refs={'prelim':p.UtilityReference(**r),'main':m.UtilityReference(**r,a_bg=abg)}
    cfgs={phase:yaml.safe_load((SOURCE_DIR/f'sources/{fn}_scoring_config.yaml').read_text()) for phase,fn in [('prelim','prelim-attack-20260825'),('main','main-process-20260912')]}
    results={}
    for phase,mod,lev in [('prelim',p,PLevel),('main',m,MLevel)]:
        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter('always')
            z=mod.score_utility(cm if phase=='main' else c,bm if phase=='main' else b,refs[phase],cfgs[phase],lev.BREAKDOWN)
        results[phase]={'score':z.score,'facets':z.facets,'detail':z.detail,'warnings':[str(w.message) for w in warns]}
        save(f'self5_{phase}_breakdown.json',results[phase]);print(phase,z.score,z.facets,'elapsed',time.time()-t,flush=True)
    ledger=pd.read_csv(SOURCE_DIR/'prelim_official_utility_all22.csv').set_index('team').loc[5]
    rr=[]
    for name in ['U','U_gen','U_spec','U_rare','U_valid']:
        old=results['prelim']['score'] if name=='U' else results['prelim']['facets'][name]
        new=results['main']['score'] if name=='U' else results['main']['facets'][name]
        rr.append({'team':5,'metric':name,'prelim_published':ledger[name],'prelim_recomputed':old,'prelim_reproduction_difference':old-ledger[name],'main_recomputed':new,'main_minus_prelim_recomputed':new-old})
    pd.DataFrame(rr).to_csv(HERE/'self5_paired_scores.csv',index=False,float_format='%.17g')
    versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','statsmodels','lifelines','formulaic','autograd']}
    locked_requirements={line.split('==')[0]:line.split('==')[1] for line in (SOURCE_DIR/'sources/main-process-20260912_requirements-kit.txt').read_text().splitlines() if '==' in line and not line.startswith('#')}
    installed={n:importlib.metadata.version(n) for n in locked_requirements}
    save('self5_run_provenance.json',{'python':platform.python_version(),'versions':versions,'all_locked_versions':installed,'all_locked_versions_match':installed==locked_requirements,'elapsed_seconds':time.time()-t,'official_locked_dependencies_used':installed==locked_requirements,'same_container_image':False,'prelim_max_abs_reproduction_error':max(abs(x['prelim_reproduction_difference']) for x in rr),'input_sha256':{str(x):sha(x) for x in [bp,cp,ap,args.reference]},'canonicalization':'Exact official adapter content-sort mergesort + record_id; B kept original record IDs and is_rare passed only through reference','counterfactual':'Preliminary B,C,reference and A_bg held fixed, main-process official code and configuration','not_actual_main_score':True})
    print(pd.DataFrame(rr).to_string(index=False),flush=True)

if __name__=='__main__':run()
