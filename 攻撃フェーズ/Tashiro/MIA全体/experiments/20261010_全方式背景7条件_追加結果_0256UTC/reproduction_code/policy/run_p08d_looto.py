"""Original small-gap 19-score panel and four-mode target-heldout selection."""
import argparse,time,json
from datetime import datetime,timezone
import numpy as np,pandas as pd
import runtime as r
p08=r.module('policy_p08',r.HERE/'original/run_p08d.py')
p08.MARGINS=(.001,.005,.01);p08.PENALTIES=(.02,.05)
p08.METHODS=('P01',)+tuple(f'sparse_top{k}_margin{m:g}_penalty{p:g}' for k in p08.TOP_K for m in p08.MARGINS for p in p08.PENALTIES)
audit=r.module('policy_p08audit',r.HERE/'original/audit_smallgap_LOOTO.py')

def run(bg):
 out=r.HERE/'cache/p08d'/bg;out.mkdir(parents=True,exist_ok=True);start=time.time();d=r.data(bg)
 indices=[j for j,t in enumerate(d['teams']) if t not in (1,2)];small=d.copy();small['teams']=[d['teams'][j] for j in indices]
 for view in audit.BG:
  p=out/view
  if (p/'prediction_record.json').exists():continue
  if view=='full':cal=r.raw_cal(bg,d)[:,indices]
  else:
   seed=int(view.split('seed')[1]);rng=np.random.default_rng(seed);aa=d['a'].iloc[np.sort(rng.choice(len(d['a']),int(.8*len(d['a'])),replace=False))]
   cols=[]
   for j in indices:
    t=d['teams'][j];raw=r.u.base.core(aa,d['query'],d['cs'][t],r.u.NUM7,(10,))[10];ctrl=d['controls'][t]
    cols.append(r.calib.calibrate(raw[:len(d['b'])],d['b'],raw[ctrl],d['query'].iloc[ctrl],['prefecture','sex'],500))
   cal=np.column_stack(cols)
  scores,stats=p08.panel(r.base,cal,d['normal'],list(range(len(indices))));assert len(scores)==19
  r.seal_scores(p,scores,small);pd.DataFrame(stats).to_csv(p/'collision_audit.csv',index=False)
  print('P08D_FROZEN',bg,view,round(time.time()-start,1),flush=True)
 r.dump(out/'freeze_record.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'truth_used_for_prediction':False,'views':audit.BG})
 if bg=='prelim':
  old=json.loads((r.SNAP/'20261009_P08D_上位競合だけの疎な排他補正/small_gap/prediction_hashes.json').read_text());checks=[]
  for x in old:
   if x['inference']!='without1_2':continue
   v=r.load_scores(out/x['background'])[x['method']];checks.append({'background':x['background'],'method':x['method'],'exact_match':r.msha(v)==x['sha256']})
  r.dump(out/'historical_prediction_regression.json',checks);assert all(x['exact_match'] for x in checks),checks
 y=r.truth(d)[:,indices];rows=[]
 for view in audit.BG:
  for method,x in r.load_scores(out/view).items():rows.extend(r.metrics_rows(x,small,y,background=view,outer_background=bg,method=method))
 by=pd.DataFrame(rows);by.to_csv(out/'base_by_target.csv',index=False,float_format='%.17g');trainpanel=by[by.population=='historical_native'];methods=list(trainpanel.method.drop_duplicates());choices=[];policy=[]
 for mode in audit.MODES:
  for target in small['teams']:
   peer=trainpanel[trainpanel.team!=target];chosen,meta=audit.choose(peer,mode,methods)
   choices.append({'outer_background':bg,'policy':mode,'team':target,'chosen_method':chosen,'training_targets':';'.join(map(str,sorted(peer.team.unique()))),**meta})
   selected=by[(by.team==target)&(by.method==chosen)].copy();selected['policy']=mode;selected['selected_method']=chosen;policy.extend(selected.to_dict('records'))
 choice=pd.DataFrame(choices);table=pd.DataFrame(policy)
 choice.to_csv(out/'selection_audit.csv',index=False);table.to_csv(out/'policy_by_target.csv',index=False,float_format='%.17g')
 summary=table.groupby(['policy','background','population'],sort=False)['tpr_0.01'].agg(targets='size',mean_tpr='mean',TPR_ge010=lambda x:int((x>=.1-1e-12).sum())).reset_index()
 summary.to_csv(out/'summary.csv',index=False,float_format='%.17g')
 if bg=='prelim':
  path=r.SNAP/'20261009_P08D_上位競合だけの疎な排他補正/looto/LOOTO_summary.csv'
  old=pd.read_csv(path)
  current=summary[(summary.population=='historical_native')&(summary.background=='full')].merge(old,on='policy')
  err=float((current.mean_tpr-current.mean_full_TPR).abs().max())
  same=bool((current.TPR_ge010==current.TPR010_hits).all())
  r.dump(out/'historical_policy_regression.json',{'rows':len(current),'max_mean_tpr_error':err,'hits_identical':same})
  assert len(current)==4 and err<1e-12 and same
 r.dump(out/'complete.json',{'seconds':time.time()-start,'outer_background':bg,'policies':4,'targets':18,'all_folds_target_excluded':True,'main_not_applicable':'original policy has no main-cohort training/application rule'})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--backgrounds',nargs='+',default=list(r.BG));a=p.parse_args()
 for bg in a.backgrounds:run(bg)
