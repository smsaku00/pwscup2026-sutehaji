from pathlib import Path
import sys,importlib.util,hashlib,json,types,ast,warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import numpy as np,pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors
HERE=Path(__file__).resolve().parent
GRID=Path('/workspace/shared/mia_all_bg_fusion/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/recovered_oct8')
AVAILABLE=Path('/workspace/shared/mia_all_bg_available/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/available')
SNAP=Path('/workspace/shared/mia_all_background_inventory/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/source_snapshot/攻撃フェーズ/Tashiro/MIA全体/experiments')
sys.path.insert(0,str(GRID));from benchmark import data,u,SRC
import optimized as opt
import 距離と分布の探索 as geo
import 分布を揃えた距離の探索 as align
import 階層ヒストグラムの探索 as hist
import 混合分布と局所密度の探索 as local
sys.path.insert(0,str(HERE/'original'))
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
base=module('policy_base',AVAILABLE/'original/base.py')
calib=module('policy_calibration',AVAILABLE/'original/非会員を使う順位較正の検討.py')
archive=module('policy_archive',HERE/'original/run_archived_kdeA_mix.py')
clinical=module('policy_clinical',HERE/'original/clinical.py')
p06=module('policy_p06',HERE/'original/run_nonmember_holdout.py')
p07=module('policy_p07',HERE/'original/run_qcohort_holdout.py')
p05=module('policy_p05',HERE/'original/run_p05.py')
BG=('prelim','starter','main','starter+prelim','starter+main','prelim+main','starter+prelim+main')
INNER=[('full',1.,0)]+[(f'fraction{f:g}_seed{s}',f,s) for f in (.9,.8,.5) for s in (20261007,20261008)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def msha(x):return hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def cross(x,w=.1):return x-w*base.rivals(x,np.median) if w else x
def cached_raw(bg,d):
 p=AVAILABLE/'results_prelim'/f'raw_cache_{bg}.npz'
 meta=json.loads(p.with_suffix('.json').read_text())
 assert meta['background_sha256']==sha(f'/workspace/shared/mia_bg7_inputs/bg/subsets/{bg}.csv')
 with np.load(p) as z:return z['raw']
def cal_from_raw(raw,d):
 b=d['b'];n=len(b);q=d['query']
 return np.column_stack([calib.calibrate(raw[:n,j],b,raw[d['controls'][t],j],q.iloc[d['controls'][t]],['prefecture','sex'],500) for j,t in enumerate(d['teams'])])
def raw_cal(bg,d):
 return cal_from_raw(cached_raw(bg,d),d)
def group_features(d):
 return clinical.build_public_features(u.base,d['b'],d['a'],d['qs'],d['cs'],d['teams'])
def assignments(groups,cuts):
 g=groups.set_index('team');return {t:('P01' if not g.loc[t,'aia_imprint_high'] else 'KDEA' if g.loc[t,'C_home_fraction']>=cuts['c_home_median'] else 'KDEB') for t in p06.TARGET17}
def truth(d):
 td=Path('/workspace/shared/mia_bg7_inputs/root/公式データ/予備戦/結果/正解データ');own=pd.read_csv(td/'pool_ids_5.csv').set_index('record_id').loc[d['b'].record_id].pool_id
 return np.column_stack([own.isin(pd.read_csv(td/f'pool_ids_{t}.csv').pool_id).to_numpy(int) for t in d['teams']])
def native_scores(x,d):
 return x if len(x)==len(d['b']) else base.confidence(x,d['normal'],d['private'],d['rare'])
def metrics_rows(x,d,y,**labels):
 native=native_scores(x,d);fixed=base.confidence(native[d['normal']],d['normal'],d['private'],d['rare'])
 rows=[]
 for pop,mat,mask in [('historical_native',native,np.ones(len(y),bool)),('all_fixed',fixed,np.ones(len(y),bool)),('normal',fixed,d['normal'])]:
  for j,t in enumerate(d['teams']):
   rows.append({**labels,'team':t,'population':pop,'n':int(mask.sum()),'members':int(y[mask,j].sum()),**base.metrics(mat[mask,j],y[mask,j])})
 return rows
def seal_scores(p,models,d,extra=None):
 p=Path(p);p.mkdir(parents=True,exist_ok=True)
 arrays={f'x{i}':v for i,v in enumerate(models.values())};np.savez_compressed(p/'scores.npz',**arrays)
 dump(p/'prediction_record.json',{'methods':list(models),'teams':d['teams'],'hashes':{k:msha(v) for k,v in models.items()},'file_sha256':sha(p/'scores.npz'),'truth_read_for_prediction':False,'extra':extra or {}})
def load_scores(p):
 p=Path(p);meta=json.loads((p/'prediction_record.json').read_text());assert sha(p/'scores.npz')==meta['file_sha256']
 with np.load(p/'scores.npz') as z:out={m:z[f'x{i}'] for i,m in enumerate(meta['methods'])}
 assert all(msha(v)==meta['hashes'][k] for k,v in out.items());return out

# Original narrow drivers need their background_common module; use an adapter
# for paths and I/O only. Mathematical helpers come from unchanged source.
bc=types.ModuleType('background_common');bc.u=u;bc.SOURCE=SRC;bc.ROOT=GRID/'original';bc.module=module
bc.HERE=HERE/'original';bc.CONDITIONS=INNER
bc.variant=lambda value,d,t,calibration='raw':u.variants({'fixed':np.asarray(value)},d,t,calibration!='raw')['fixed__'+calibration]
bc.cross=lambda bank,d,coefficient:cross(np.column_stack([bank[t] for t in d['teams']]),coefficient)
sys.modules['background_common']=bc
geometry=module('policy_geometry',HERE/'original/geometry_robustness.py')
histlocal=module('policy_histlocal',HERE/'original/hist_local_robustness.py')
legacy=module('policy_legacy',HERE/'original/legacy_robustness.py')
public=module('policy_public',HERE/'original/公開加工特徴.py')

def narrow_geometry(d):
 prepared={}
 for attrs,coord,mults in [('q7','robust',(.5,1.)),('q7','rank',(.5,)),('q13','robust',(.5,2.))]:
  nums=u.NUM7 if attrs=='q7' else u.NUM13;cats=u.CAT7 if attrs=='q7' else u.CAT13;q=d['query'] if attrs=='q7' else d['b'];a=d['a']
  ar,x=geo.transform(a,[a,q],nums,coord);h0=(4/(len(nums)+2)/len(a))**(1/(len(nums)+4))
  den={w:opt.mixed_chunked(x,ar,q,a,cats,a,h0*w) for w in mults}
  prepared[attrs,coord]=dict(nums=nums,cats=cats,query=q,x=x,h0=h0,coordinate=coord,density=den)
 banks={s['method']:{} for s in geometry.SPECS};nn=geometry.original_nn_reference(d)
 for t in d['teams']:
  banks['global_nn_k1'][t]=bc.variant(geometry.original_nn_score(d,d['cs'][t],nn),d,t,'raw')
  for s in geometry.SPECS[1:]:
   v=geometry.fixed_score(d,d['cs'][t],prepared[s['attributes'],s['coordinate']],s)
   banks[s['method']][t]=bc.variant(v,d,t,s['calibration'])
 return {s['method']:bc.cross(banks[s['method']],d,s['cross']) for s in geometry.SPECS}

def narrow_histlocal(d,only_kdeb=False):
 settings=[s for s in histlocal.SETTINGS if not only_kdeb or s[0]=='q7_robust_adaptivek15_h1']
 banks={s[0]+'__'+s[1]:{} for s in settings};a=d['a'];prepared={};den={};lofa={}
 views=[('q7','robust')] if only_kdeb else [('q7','robust'),('q7','rank'),('q13','robust')]
 for view,kind in views:
  nums=u.NUM7 if view=='q7' else u.NUM13;cats=u.CAT7 if view=='q7' else u.CAT13;q=d['query'] if view=='q7' else d['b']
  ar,x=geo.transform(a,[a,q],nums,kind);prepared[view,kind]=(ar,x,q,nums,cats)
  if kind=='rank':
   f=local.LocalOutlierFactor(n_neighbors=100,novelty=True,n_jobs=1);f.fit(ar);lofa[view,kind]=f.score_samples(x)
  else:
   k,w=(15,1.) if view=='q7' else (5,.5);ka=min(len(a)-1,round(k*len(a)/1049))
   scale=opt.anchor_scales(ar,[ka],len(nums))[ka];den[view,kind]=opt.adaptive(x,ar,q,a,a,cats,scale,w,len(nums))
 for t in d['teams']:
  c=d['cs'][t];raw={}
  if not only_kdeb:
   for view,group,bins in [('q7','all',20),('q7','sex',40),('q13','pref_sex',10)]:
    nums=u.NUM7 if view=='q7' else u.NUM13;cats=u.CAT7 if view=='q7' else u.CAT13;q=d['query'] if view=='q7' else d['b']
    cols=[] if group=='all' else ['sex'] if group=='sex' else ['prefecture','sex'];vals=hist.histograms(a,q,c,nums,cats,bins,cols)
    for s in settings:
     if s[4]=='hist' and (s[5],s[6],s[7])==(view,group,bins):raw[s[0]]=vals[s[8]]
  for view,kind in views:
   ar,x,q,nums,cats=prepared[view,kind];cr=geo.transform(a,[c],nums,kind)[0]
   if not only_kdeb:
    f=local.LocalOutlierFactor(n_neighbors=100,novelty=True,n_jobs=1);f.fit(cr);v=f.score_samples(x);raw[f'{view}_{kind}_lofC_k100']=v
    if kind=='rank':raw[f'{view}_{kind}_lof_ratio_k100']=v-lofa[view,kind]
   if kind!='rank':
    k,w=(15,1.) if view=='q7' else (5,.5);scale=opt.anchor_scales(cr,[k],len(nums))[k]
    raw[f'{view}_{kind}_adaptivek{k}_h{w:g}']=opt.adaptive(x,cr,q,c,a,cats,scale,w,len(nums))-den[view,kind]
  for s in settings:
   stem,cal=s[:2];key=stem+'__'+cal;banks[key][t]=u.variants({stem:raw[stem]},d,t,s[5]=='q7')[key]
 return {histlocal.method_name(s):u.matrices({s[0]+'__'+s[1]:banks[s[0]+'__'+s[1]]},d['teams'],cross=(s[2],))[histlocal.method_name(s)] for s in settings}
