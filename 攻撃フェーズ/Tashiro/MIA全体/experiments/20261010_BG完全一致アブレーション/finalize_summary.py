"""Complete aggregate output after a scoring-only summary-column correction."""
from pathlib import Path
from datetime import datetime,timezone
import platform,sys
import pandas as pd,numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import run_ablation as run
out=HERE/'starter+prelim';df=pd.read_csv(out/'by_condition.csv',float_precision='round_trip');summaries=[]
for (t,method,pop),g in df.groupby(['team','method','population'],sort=False):
    orig=g[g.kind=='original'].iloc[0];exact=g[g.kind=='exact'].iloc[0];rnd=g[g.kind=='random']
    for metric in ('tpr_0.01','auc','top25_expected_hits'):
        v=rnd[metric].to_numpy();delta=float(exact[metric]-orig[metric])
        summaries.append(dict(team=t,method=method,population=pop,metric=metric,original=float(orig[metric]),remove_exact=float(exact[metric]),delta_exact=delta,random_mean=float(v.mean()),random_min=float(v.min()),random_max=float(v.max()),random_sd=float(v.std(ddof=1)),excess_over_random_mean=float(exact[metric]-v.mean()),removed=int(exact.removed),random_repeats=len(v)))
pd.DataFrame(summaries).to_csv(out/'ablation_summary.csv',index=False,float_format='%.17g')
run.dump(out/'completion.json',{'finished_utc':datetime.now(timezone.utc).isoformat(),'condition_count':19,'rows':len(df),'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scoring_only_correction':'../scoring_only_correction.json','finisher_sha256':run.sha(__file__)})
print(pd.DataFrame(summaries).query("population == 'all_fixed' and metric == 'tpr_0.01'").to_string(index=False))
